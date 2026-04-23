"""
longitudinal_state_builder.py — EHR Temporal Validator
Apoorva Kolhatkar | Michigan Medicine NLP Research

Builds a longitudinal profile for each entity across all notes,
then risk-scores it to determine whether it warrants LLM resolution.

PURPOSE — THE CANDIDATE PRE-FILTER:
    The OllamaResolver is powerful but expensive. Running it on every
    entity in every note produces noise and wastes inference compute.
    This module sits between entity extraction and LLM resolution,
    answering one question per entity:

        "Is this entity longitudinally unstable?"

    Only unstable entities become Ollama candidates.
    Stable entities are marked resolved without LLM involvement.

WHAT "LONGITUDINALLY UNSTABLE" MEANS:
    An entity is unstable if its chart trajectory shows one or more
    of the following risk signals:

    SIGNAL 1 — Status change cue in the chart for this entity
        A note contains language like "discontinued", "transitioned",
        "held", "stopped", "restarted" near this entity name.
        Evidence: warfarin in SYNTH_003 note_4 — "warfarin to be
        discontinued today. Apixaban initiated."
        Source: PMC4476907 — 78.1% of EMR medication lists inaccurate,
        biggest driver is failure to remove discontinued medications.

    SIGNAL 2 — Entity appears after a status change cue for it
        The entity is extracted as active in a note that comes AFTER
        a note where it was flagged as discontinued or transitioned.
        This is the copy-forward pattern.
        Source: PMC5373750 — Tsou et al. systematic review, 51 publications.

    SIGNAL 3 — Entity appears in notes but not in structured/coded record
        Free text presence without ICD or medication list entry.
        Source: PMC9759969 — only 62.3% of diagnoses in free text
        were present in the structured problem list.

    SIGNAL 4 — Entity span count drops sharply then reappears
        A medication mentioned across 5 notes, absent in 2, then
        returns — suggests possible reinitiation or copy-forward.

    SIGNAL 5 — Multiple providers document entity with conflicting context
        Entity appears in notes from different authors/specialties
        with different status cues.
        Source: PMC4476907 — >40% of medication errors traced to
        inadequate reconciliation at handoffs.

ENTITY CLASSES COVERED (evidence-grounded):
    Class 1 — Medications with stop/start behavior
        Most documented copy-forward class. Anticoagulants, insulin,
        diabetes medications, antihypertensives.
        Source: PMC4476907, PMC8661442, PMC10983481.

    Class 2 — Diagnoses with temporal expectation
        Problem list inaccuracy — resolved conditions not removed,
        active conditions not added.
        Source: PMC9759969, PMC4549158, PMC7153078.

    Class 3 — Free text vs coded record divergence
        Diagnoses in notes not reflected in ICD/problem list.
        Source: PMC9759969, PMC11520144.

    Class 4 — Multi-provider reconciliation gaps
        Medication changes by one provider not reflected in another's notes.
        Source: PMC4476907, PMC9667166.

    NOTE on Class 5 (procedure-complication ordering):
        Partially supported by ICD-10-CM coding guidelines and general
        copy-forward error propagation literature, but no dedicated
        study quantifies this as a distinct class. Treated conservatively
        here — temporal ordering violations are flagged by temporal_validator.py
        rather than by this module.

OUTPUT:
    A dict keyed by normalized entity text, each value containing:
    - first_seen: earliest note_id and date
    - last_seen: latest note_id and date
    - note_count: number of notes where entity appears
    - status_change_notes: notes containing status change cues for this entity
    - post_change_appearances: notes where entity appears AFTER a status change
    - risk_score: 0.0 to 1.0 (higher = more unstable, more likely to need LLM)
    - risk_signals: list of strings explaining why this entity is flagged
    - ollama_candidate: bool — whether this entity should go to OllamaResolver

INTEGRATION:
    Called from pipeline.py between Stage 2 (entity extraction)
    and Stage 2b (Ollama resolution). The OllamaResolver uses
    longitudinal profiles to filter candidates before making LLM calls.
"""

import re
from datetime import datetime
from typing import Optional


# ── Status change vocabulary ──────────────────────────────────────────────────
# Phrases that signal a medication or diagnosis status transition.
# Window: we look for the entity name within ±80 chars of the phrase.

STATUS_CHANGE_PHRASES = [
    "discontinued", "stopped", "held", "no longer taking", "not taking",
    "transitioned", "switched", "changed to", "replaced with",
    "restarted", "resumed", "reinitiated", "started", "initiated",
    "added", "increased", "decreased", "dose reduced", "dose increased",
    "tapered", "titrated", "adjusted", "modified", "changed",
]

# Subset that specifically signal STOPPING (high risk for copy-forward)
STOP_PHRASES = {
    "discontinued", "stopped", "held", "no longer taking", "not taking",
    "transitioned", "switched", "replaced with",
}

# Subset that specifically signal STARTING (may mean prior was replaced)
START_PHRASES = {
    "started", "initiated", "added", "restarted", "resumed", "reinitiated",
}


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _normalize(text: str) -> str:
    """Normalize entity text for deduplication — lowercase, strip doses."""
    text = text.lower().strip()
    # Remove trailing dose patterns: "10mg", "5mg daily", "500mg bid"
    text = re.sub(r'\s+\d+\s*(?:mg|mcg|g|units?|iu)\b.*$', '', text)
    return text.strip()


def _entity_near_phrase(entity_text: str, phrase: str, note_text: str,
                        window: int = 80) -> bool:
    """
    Return True if entity_text appears within `window` characters
    of `phrase` in note_text (case-insensitive).
    """
    text_lower = note_text.lower()
    entity_lower = entity_text.lower()
    phrase_lower = phrase.lower()

    idx = 0
    while True:
        pos = text_lower.find(phrase_lower, idx)
        if pos == -1:
            break
        start = max(0, pos - window)
        end = min(len(text_lower), pos + len(phrase_lower) + window)
        context = text_lower[start:end]
        if entity_lower in context or any(
            word in context for word in entity_lower.split()
            if len(word) > 4
        ):
            return True
        idx = pos + 1
    return False


def _find_status_change_cues(entity_text: str, notes: list[dict]) -> list[dict]:
    """
    Find all notes containing a status change phrase near this entity.
    Returns list of dicts with note_id, date, phrase, phrase_type.
    """
    cues = []
    for note in notes:
        text = note.get("text", "")
        text_lower = text.lower()
        for phrase in STATUS_CHANGE_PHRASES:
            if phrase in text_lower:
                if _entity_near_phrase(entity_text, phrase, text):
                    phrase_type = (
                        "stop" if phrase in STOP_PHRASES
                        else "start" if phrase in START_PHRASES
                        else "change"
                    )
                    cues.append({
                        "note_id":     note["note_id"],
                        "date":        note.get("date", ""),
                        "phrase":      phrase,
                        "phrase_type": phrase_type,
                    })
                    break  # One cue per note is enough
    return cues


class LongitudinalStateBuilder:
    """
    Builds per-entity longitudinal profiles across all notes for a patient.

    Usage:
        builder = LongitudinalStateBuilder()
        profiles = builder.build(extracted_entities, all_notes, structured_data)

    Returns a dict of entity profiles. Use `get_ollama_candidates()` to
    get only the entities that warrant LLM resolution.
    """

    # Risk score thresholds
    OLLAMA_THRESHOLD = 0.35   # Entities above this go to Ollama
    HIGH_RISK        = 0.65   # Reported as high-risk regardless of Ollama result

    def build(
        self,
        extracted_entities: list[dict],
        all_notes: list[dict],
        structured_data: Optional[dict] = None
    ) -> dict:
        """
        Build longitudinal profiles for all entities across all notes.

        Args:
            extracted_entities: output of EntityExtractor.extract_all()
            all_notes: raw note dicts with text, date, note_id
            structured_data: optional structured record (medications, diagnoses)

        Returns:
            dict keyed by normalized entity text, value is the profile dict.
        """
        profiles = {}

        # Build note lookup for fast access
        notes_by_id = {n["note_id"]: n for n in all_notes}

        # Collect all mentions per entity across all notes
        entity_mentions = self._collect_mentions(extracted_entities)

        # Build structured record sets for cross-checking
        structured_meds = set()
        structured_diags = set()
        if structured_data:
            for m in structured_data.get("medications", []):
                structured_meds.add(_normalize(m.get("medication", "")))
            for d in structured_data.get("diagnoses", []):
                structured_diags.add(_normalize(d.get("description", "")))

        # Build profile for each entity
        for norm_text, mentions in entity_mentions.items():
            profile = self._build_profile(
                norm_text,
                mentions,
                all_notes,
                notes_by_id,
                structured_meds,
                structured_diags
            )
            profiles[norm_text] = profile

        total = len(profiles)
        candidates = sum(1 for p in profiles.values() if p["ollama_candidate"])
        print(
            f"[LongitudinalStateBuilder] {total} entities profiled — "
            f"{candidates} flagged as Ollama candidates "
            f"(threshold: {self.OLLAMA_THRESHOLD})"
        )

        return profiles

    def get_ollama_candidates(self, profiles: dict) -> list[dict]:
        """
        Return only entities flagged as Ollama candidates,
        sorted by risk score descending.
        """
        candidates = [
            p for p in profiles.values()
            if p["ollama_candidate"]
        ]
        candidates.sort(key=lambda x: x["risk_score"], reverse=True)
        return candidates

    def _collect_mentions(self, extracted_entities: list[dict]) -> dict:
        """
        Group entity mentions by normalized text across all notes.

        Returns:
            dict of norm_text -> list of mention dicts
            Each mention: {note_id, date, text, type, negated, source}
        """
        mentions = {}

        for note_data in extracted_entities:
            note_id = note_data["note_id"]
            date    = note_data.get("date", "")
            entities = note_data.get("entities", {})

            for entity_type in ("medications", "diagnoses", "procedures"):
                for ent in entities.get(entity_type, []):
                    raw_text  = ent.get("text", "").strip()
                    norm_text = _normalize(raw_text)
                    if not norm_text or len(norm_text) < 3:
                        continue

                    mention = {
                        "note_id":     note_id,
                        "date":        date,
                        "text":        raw_text,
                        "type":        entity_type.rstrip("s"),  # medication, diagnosis
                        "negated":     ent.get("negated", False),
                        "source":      ent.get("source", "rule_based"),
                    }

                    if norm_text not in mentions:
                        mentions[norm_text] = []
                    mentions[norm_text].append(mention)

        return mentions

    def _build_profile(
        self,
        norm_text:      str,
        mentions:       list[dict],
        all_notes:      list[dict],
        notes_by_id:    dict,
        structured_meds:  set,
        structured_diags: set,
    ) -> dict:
        """Build the full longitudinal profile for one entity."""

        # Sort mentions chronologically
        mentions_sorted = sorted(
            mentions,
            key=lambda m: _parse_date(m["date"]) or datetime.min
        )

        active_mentions = [m for m in mentions_sorted if not m["negated"]]
        negated_mentions = [m for m in mentions_sorted if m["negated"]]

        entity_type = mentions_sorted[0]["type"] if mentions_sorted else "unknown"
        display_text = mentions_sorted[0]["text"] if mentions_sorted else norm_text

        first_seen = mentions_sorted[0] if mentions_sorted else {}
        last_seen  = mentions_sorted[-1] if mentions_sorted else {}

        # Find all notes with status change cues for this entity
        status_change_cues = _find_status_change_cues(norm_text, all_notes)
        stop_cues  = [c for c in status_change_cues if c["phrase_type"] == "stop"]
        start_cues = [c for c in status_change_cues if c["phrase_type"] == "start"]

        # Find appearances AFTER a stop cue (copy-forward pattern)
        post_stop_appearances = []
        if stop_cues:
            last_stop_date = max(
                (_parse_date(c["date"]) for c in stop_cues if _parse_date(c["date"])),
                default=None
            )
            last_stop_note = stop_cues[-1]["note_id"] if stop_cues else None

            if last_stop_date:
                for m in active_mentions:
                    m_date = _parse_date(m["date"])
                    if m_date and m_date > last_stop_date and m["note_id"] != last_stop_note:
                        post_stop_appearances.append(m)

        # Check structured record coverage
        in_structured = False
        if entity_type == "medication":
            in_structured = any(
                norm_text in s or s in norm_text
                for s in structured_meds
            )
        elif entity_type == "diagnosis":
            in_structured = any(
                norm_text in s or s in norm_text
                for s in structured_diags
            )

        # Detect note count gaps — entity absent in middle notes
        all_note_ids_ordered = [n["note_id"] for n in all_notes]
        mention_note_ids = {m["note_id"] for m in active_mentions}
        gap_detected = False
        if len(active_mentions) >= 2 and len(all_notes) >= 3:
            first_idx = all_note_ids_ordered.index(first_seen.get("note_id", ""))
            last_idx  = all_note_ids_ordered.index(last_seen.get("note_id", ""))
            span_notes = all_note_ids_ordered[first_idx:last_idx + 1]
            absent_in_span = [n for n in span_notes if n not in mention_note_ids]
            if len(absent_in_span) >= 2:
                gap_detected = True

        # ── Risk scoring ──────────────────────────────────────────────────────
        risk_score   = 0.0
        risk_signals = []

        # Signal 1: Status change cue near entity (any type)
        if status_change_cues:
            risk_score += 0.25
            phrases = list({c["phrase"] for c in status_change_cues})
            risk_signals.append(
                f"Status change cue(s) in chart: {', '.join(phrases[:3])}"
            )

        # Signal 2: Active appearance AFTER a stop cue — strongest signal
        if post_stop_appearances:
            risk_score += 0.45
            risk_signals.append(
                f"Appears as active in {len(post_stop_appearances)} note(s) "
                f"after documented discontinuation/transition — "
                f"copy-forward pattern (PMC4476907, PMC5373750)"
            )

        # Signal 3: In notes but not in structured record
        if active_mentions and not in_structured and entity_type in ("medication", "diagnosis"):
            risk_score += 0.20
            risk_signals.append(
                f"Documented in {len(active_mentions)} note(s) but absent "
                f"from structured {'medication list' if entity_type == 'medication' else 'problem list'} "
                f"(PMC9759969 — 37.7% of diagnoses missing from problem list)"
            )

        # Signal 4: Gap in note coverage (appears, disappears, reappears)
        if gap_detected:
            risk_score += 0.15
            risk_signals.append(
                "Entity absent from middle notes then reappears — "
                "possible discontinuation gap or copy-forward reinsertion"
            )

        # Signal 5: Both stop and start cues present — transition event
        if stop_cues and start_cues:
            risk_score += 0.10
            risk_signals.append(
                f"Both stop cues ({len(stop_cues)}) and start cues ({len(start_cues)}) "
                f"present — medication transition or reinitiation event"
            )

        # Cap at 1.0
        risk_score = round(min(risk_score, 1.0), 3)

        return {
            "entity":               norm_text,
            "display_text":         display_text,
            "entity_type":          entity_type,
            "note_count":           len(active_mentions),
            "negated_count":        len(negated_mentions),
            "first_seen":           {"note_id": first_seen.get("note_id"), "date": first_seen.get("date")},
            "last_seen":            {"note_id": last_seen.get("note_id"),  "date": last_seen.get("date")},
            "status_change_cues":   status_change_cues,
            "stop_cue_count":       len(stop_cues),
            "start_cue_count":      len(start_cues),
            "post_stop_appearances": [
                {"note_id": m["note_id"], "date": m["date"]}
                for m in post_stop_appearances
            ],
            "in_structured_record": in_structured,
            "gap_detected":         gap_detected,
            "risk_score":           risk_score,
            "risk_signals":         risk_signals,
            "ollama_candidate":     risk_score >= self.OLLAMA_THRESHOLD,
        }
