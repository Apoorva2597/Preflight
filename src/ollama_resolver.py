"""
ollama_resolver.py — Preflight
Apoorva Kolhatkar | Michigan Medicine NLP Research

LLM-based longitudinal entity status resolution using local Ollama inference.
Implements LLMResolver — see llm_resolver.py for the interface contract.

PURPOSE:
    NER (even GLiNER) extracts entity spans and surface-level status cues.
    It cannot reliably resolve longitudinal status — whether a medication
    mentioned in note_6 is currently active given a discontinuation documented
    in note_4. This requires reading context across multiple notes and reasoning
    about temporal order, clinical intent, and documentation patterns.

    This module handles that reasoning. For each medication and diagnosis
    extracted from the current note, it:
      1. Retrieves the most relevant prior notes via recency + keyword relevance
         (RAG retrieval layer — lightweight, no embedding model needed at this stage)
      2. Constructs a clinical prompt with the retrieved context
      3. Queries a local Ollama model for status resolution
      4. Returns structured status decisions with confidence and reasoning

ARCHITECTURE:
    This is Stage 2b in the pipeline — runs after GLiNER extracts entities
    from all notes, before confidence scoring and contradiction detection.

    Entity extraction (GLiNER)
        ↓
    [THIS MODULE] Status resolution (Ollama + RAG)
        ↓
    Confidence scoring, contradiction detection, ICD divergence

    This class implements LLMResolver. To swap in a different LLM backend
    (Claude API, Azure OpenAI, fine-tuned model), subclass LLMResolver
    and pass the new instance to main.py's lifespan. No pipeline code changes.

RAG DESIGN:
    For each entity being resolved, we retrieve:
    - The note where the entity first appeared
    - The note most proximal to the current note (temporal recency)
    - Any note containing a status-change phrase for this entity
      (discontinued, held, stopped, restarted, changed, transitioned)

    This is keyword-based retrieval — fast, deterministic, no embedding model.
    Embedding-based retrieval (MedCPT) is the planned upgrade path documented
    in the README architecture section.

OLLAMA SETUP:
    Install: https://ollama.com (one-command install)
    Pull model: ollama pull llama3.2
    Run server: ollama serve  (usually auto-starts)
    Default endpoint: http://localhost:11434

    Recommended models (in order of capability vs speed):
      llama3.2        — fast, good clinical reasoning, recommended
      llama3.1:8b     — stronger reasoning, slower
      mistral         — alternative, good instruction following
      gemma2          — compact, fast

    Clinical prompt is model-agnostic. Tested on llama3.2 and mistral.

DATA GOVERNANCE:
    Ollama runs entirely locally. No data leaves your machine.
    Compatible with PhysioNet MIMIC Data Use Agreement which prohibits
    sending credentialed data to third-party API services.
    Do not configure this module to use cloud Ollama endpoints with MIMIC data.

GRACEFUL DEGRADATION:
    If Ollama is not running or not installed:
    - The resolver returns without modifying extracted entities
    - A warning is logged
    - The pipeline continues with GLiNER-extracted status labels only
    - No crash, no silent failure
"""

import json
import re
import urllib.request
import urllib.error
import time
from typing import Optional

from llm_resolver import LLMResolver


# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "llama3.2"           # Change to llama3.1:8b for stronger reasoning
OLLAMA_TIMEOUT    = 180                   # Seconds per request
MAX_CONTEXT_NOTES = 3                    # Max prior notes passed to LLM per entity
MAX_NOTE_CHARS    = 800                  # Truncate long notes to this length for context

# Status change phrases — used for RAG retrieval to find relevant prior notes
STATUS_CHANGE_PHRASES = [
    "discontinued", "stopped", "held", "no longer taking", "not taking",
    "transitioned", "switched", "changed to", "replaced with", "restarted",
    "resumed", "reinitiated", "started", "initiated", "added",
    "increased", "decreased", "dose reduced", "dose increased",
]


# ── Ollama connectivity ───────────────────────────────────────────────────────

def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _ollama_generate(prompt: str, base_url: str = OLLAMA_BASE_URL,
                     model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Send a prompt to Ollama and return the response text.
    Uses the /api/generate endpoint with stream=False.

    Retry logic: up to 3 attempts with exponential backoff (2s, 4s, 8s).
    Handles transient Ollama hangs without failing the full pipeline.
    Returns None only after all retries are exhausted — pipeline continues
    with GLiNER-only status labels, no crash.
    """

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 200,
        }
    }).encode("utf-8")

    max_attempts = 3
    backoff = 2  # seconds — doubles each retry: 2s, 4s, 8s

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()

        except urllib.error.URLError as e:
            if attempt < max_attempts:
                print(f"[OllamaResolver] Attempt {attempt} failed ({e}). "
                      f"Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"[OllamaResolver] All {max_attempts} attempts failed. "
                      f"Continuing without LLM resolution for this entity.")
                return None

        except Exception as e:
            print(f"[OllamaResolver] Unexpected error on attempt {attempt}: {e}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
            else:
                return None


# ── RAG retrieval layer ───────────────────────────────────────────────────────

def _retrieve_relevant_notes(
    entity_text: str,
    current_note_id: str,
    all_notes: list[dict],
    max_notes: int = MAX_CONTEXT_NOTES
) -> list[dict]:
    """
    Retrieve the most relevant prior notes for a given entity.

    Retrieval strategy (keyword-based, no embeddings):
      Priority 1 — Notes containing a status-change phrase for this entity
                   (stopped, discontinued, transitioned, restarted...)
      Priority 2 — Notes containing the entity name
      Priority 3 — Most recent prior notes (temporal recency)

    Returns notes in chronological order, truncated to MAX_NOTE_CHARS.
    """
    entity_lower = entity_text.lower().strip()
    prior_notes = [
        n for n in all_notes
        if n["note_id"] != current_note_id
    ]

    scored = []
    for note in prior_notes:
        text_lower = note.get("text", "").lower()
        score = 0

        has_entity = entity_lower in text_lower or any(
            word in text_lower for word in entity_lower.split()
            if len(word) > 4
        )
        has_status_change = any(
            phrase in text_lower for phrase in STATUS_CHANGE_PHRASES
            if entity_lower in text_lower[
                max(0, text_lower.find(phrase) - 60):
                text_lower.find(phrase) + 60
            ] if phrase in text_lower
        )

        if has_entity and has_status_change:
            score = 3
        elif has_entity:
            score = 2
        elif has_status_change:
            score = 1

        scored.append((score, note.get("date", ""), note))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [item[2] for item in scored[:max_notes]]
    selected.sort(key=lambda n: n.get("date", ""))

    return [
        {
            "note_id":  n["note_id"],
            "date":     n.get("date", "unknown"),
            "category": n.get("category", ""),
            "text":     n.get("text", "")[:MAX_NOTE_CHARS] + (
                "..." if len(n.get("text", "")) > MAX_NOTE_CHARS else ""
            )
        }
        for n in selected
    ]


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_medication_prompt(
    medication: str,
    current_note: dict,
    prior_notes: list[dict]
) -> str:
    prior_context = ""
    if prior_notes:
        prior_context = "\n\nRELEVANT PRIOR NOTES:\n"
        for n in prior_notes:
            prior_context += (
                f"\n[{n['date']} — {n['category']} — {n['note_id']}]\n"
                f"{n['text']}\n"
            )
    else:
        prior_context = "\n\nNo prior notes available for context.\n"

    current_text = current_note.get("text", "")[:MAX_NOTE_CHARS]
    current_date = current_note.get("date", "unknown")
    current_id   = current_note.get("note_id", "unknown")

    return f"""You are a clinical informatics system analyzing an electronic health record.

Your task: Determine the current status of the medication "{medication}" as of the note below.

CURRENT NOTE ({current_date} — {current_id}):
{current_text}
{prior_context}

Based on the notes above, answer these questions about "{medication}":

1. STATUS: Is this medication currently ACTIVE, DISCONTINUED, HELD, or UNCERTAIN?
2. CONFIDENCE: HIGH, MEDIUM, or LOW
3. REASON: One sentence explaining your determination.
4. COPY_FORWARD_RISK: YES or NO — does the current note appear to be copying forward a prior status without reflecting a documented change?

Respond in this exact format:
STATUS: [ACTIVE|DISCONTINUED|HELD|UNCERTAIN]
CONFIDENCE: [HIGH|MEDIUM|LOW]
REASON: [one sentence]
COPY_FORWARD_RISK: [YES|NO]"""


def _build_diagnosis_prompt(
    diagnosis: str,
    current_note: dict,
    prior_notes: list[dict]
) -> str:
    prior_context = ""
    if prior_notes:
        prior_context = "\n\nRELEVANT PRIOR NOTES:\n"
        for n in prior_notes:
            prior_context += (
                f"\n[{n['date']} — {n['category']} — {n['note_id']}]\n"
                f"{n['text']}\n"
            )
    else:
        prior_context = "\n\nNo prior notes available for context.\n"

    current_text = current_note.get("text", "")[:MAX_NOTE_CHARS]
    current_date = current_note.get("date", "unknown")
    current_id   = current_note.get("note_id", "unknown")

    return f"""You are a clinical informatics system analyzing an electronic health record.

Your task: Determine the current status of the diagnosis "{diagnosis}" as of the note below.

CURRENT NOTE ({current_date} — {current_id}):
{current_text}
{prior_context}

Based on the notes above, answer these questions about "{diagnosis}":

1. STATUS: Is this diagnosis ACTIVE, RESOLVED, HISTORICAL, NEW, or UNCERTAIN?
2. CONFIDENCE: HIGH, MEDIUM, or LOW
3. REASON: One sentence explaining your determination.
4. CODED: Based on the notes, does this diagnosis appear to be present in the structured/coded record, or only in free text?

Respond in this exact format:
STATUS: [ACTIVE|RESOLVED|HISTORICAL|NEW|UNCERTAIN]
CONFIDENCE: [HIGH|MEDIUM|LOW]
REASON: [one sentence]
CODED: [YES|NO|UNCERTAIN]"""


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_medication_response(response: str, medication: str) -> dict:
    """Parse structured LLM response for medication status."""
    result = {
        "entity":              medication,
        "entity_type":         "medication",
        "status":              "UNCERTAIN",
        "confidence":          "LOW",
        "reason":              "",
        "copy_forward_risk":   False,
        "resolved_by":         "ollama",
        "model":               OLLAMA_MODEL,
    }
    if not response:
        return result

    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("STATUS:"):
            val = line.replace("STATUS:", "").strip().upper()
            if val in ("ACTIVE", "DISCONTINUED", "HELD", "UNCERTAIN"):
                result["status"] = val
        elif line.startswith("CONFIDENCE:"):
            val = line.replace("CONFIDENCE:", "").strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                result["confidence"] = val
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("COPY_FORWARD_RISK:"):
            val = line.replace("COPY_FORWARD_RISK:", "").strip().upper()
            result["copy_forward_risk"] = val == "YES"

    return result


def _parse_diagnosis_response(response: str, diagnosis: str) -> dict:
    """Parse structured LLM response for diagnosis status."""
    result = {
        "entity":       diagnosis,
        "entity_type":  "diagnosis",
        "status":       "UNCERTAIN",
        "confidence":   "LOW",
        "reason":       "",
        "coded":        "UNCERTAIN",
        "resolved_by":  "ollama",
        "model":        OLLAMA_MODEL,
    }
    if not response:
        return result

    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("STATUS:"):
            val = line.replace("STATUS:", "").strip().upper()
            if val in ("ACTIVE", "RESOLVED", "HISTORICAL", "NEW", "UNCERTAIN"):
                result["status"] = val
        elif line.startswith("CONFIDENCE:"):
            val = line.replace("CONFIDENCE:", "").strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                result["confidence"] = val
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("CODED:"):
            val = line.replace("CODED:", "").strip().upper()
            if val in ("YES", "NO", "UNCERTAIN"):
                result["coded"] = val

    return result


# ── Main resolver ─────────────────────────────────────────────────────────────

class OllamaResolver(LLMResolver):
    """
    Longitudinal entity status resolver using local Ollama inference.
    Implements the LLMResolver interface.

    To swap in a different LLM backend, subclass LLMResolver and implement
    resolve_all() and resolve_single() — no pipeline code changes needed.

    Usage:
        resolver = OllamaResolver()
        enriched = resolver.resolve_all(extracted_entities, all_notes)

    The resolver adds a "resolved_statuses" key to each note's entity dict.
    Downstream modules (named_flags, contradiction detection) use this.
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model     = model
        self.base_url  = base_url
        self._enabled  = False

        if _ollama_available():
            self._enabled = True
            print(f"[OllamaResolver] Connected — model: {model}")
        else:
            print(
                "[OllamaResolver] Ollama not available. "
                "Install from https://ollama.com and run: ollama pull llama3.2 "
                "Pipeline continues with GLiNER status labels only."
            )

    def resolve_all(
        self,
        extracted_entities: list[dict],
        all_notes: list[dict],
        candidate_profiles: Optional[dict] = None,
    ) -> list[dict]:
        """
        Enrich extracted entities with LLM-resolved status.

        When candidate_profiles is provided (from LongitudinalStateBuilder),
        only entities flagged as ollama_candidate=True are resolved.
        This is the critical pre-filter — prevents wasted inference on
        stable entities with no longitudinal inconsistency.

        If Ollama is unavailable, returns extracted_entities unchanged.
        """
        if not self._enabled:
            return extracted_entities

        notes_by_id = {n["note_id"]: n for n in all_notes}

        candidate_keys = None
        if candidate_profiles:
            candidate_keys = {
                key for key, profile in candidate_profiles.items()
                if profile.get("ollama_candidate", False)
            }
            print(
                f"[OllamaResolver] Candidate filter active — "
                f"{len(candidate_keys)} entities flagged for resolution."
            )

        total_resolved = 0
        total_skipped  = 0

        for note_data in extracted_entities:
            note_id      = note_data["note_id"]
            entities     = note_data.get("entities", {})
            current_note = notes_by_id.get(note_id, {})
            resolved     = []

            for entity_type, prompt_fn, parse_fn in [
                ("medications", _build_medication_prompt, _parse_medication_response),
                ("diagnoses",   _build_diagnosis_prompt,  _parse_diagnosis_response),
            ]:
                for ent in entities.get(entity_type, []):
                    if ent.get("negated"):
                        continue

                    ent_text = ent["text"]
                    norm     = ent_text.lower().strip()
                    norm     = re.sub(r"\s+\d+\s*(?:mg|mcg|g|units?|iu).*$", "", norm).strip()

                    if candidate_keys is not None:
                        is_candidate = any(
                            norm in ck or ck in norm
                            for ck in candidate_keys
                        )
                        if not is_candidate:
                            total_skipped += 1
                            continue

                    prior_notes = _retrieve_relevant_notes(ent_text, note_id, all_notes)
                    prompt      = prompt_fn(ent_text, current_note, prior_notes)
                    response    = _ollama_generate(prompt, self.base_url, self.model)
                    decision    = parse_fn(response, ent_text)
                    decision["note_id"]              = note_id
                    decision["retrieved_note_ids"]   = [n["note_id"] for n in prior_notes]
                    resolved.append(decision)
                    total_resolved += 1

            note_data["entities"]["resolved_statuses"] = resolved

        print(
            f"[OllamaResolver] Resolved {total_resolved} entities "
            f"({total_skipped} skipped — stable by longitudinal profile)."
        )
        return extracted_entities

    def resolve_single(
        self,
        entity_text: str,
        entity_type: str,
        current_note: dict,
        all_notes: list[dict],
    ) -> Optional[dict]:
        """
        Resolve a single entity. Useful for targeted re-evaluation
        or interactive debugging without running the full pipeline.

        entity_type: "medication" or "diagnosis"
        """
        if not self._enabled:
            return None

        prior_notes = _retrieve_relevant_notes(
            entity_text, current_note["note_id"], all_notes
        )

        if entity_type == "medication":
            prompt   = _build_medication_prompt(entity_text, current_note, prior_notes)
            response = _ollama_generate(prompt, self.base_url, self.model)
            return _parse_medication_response(response, entity_text)
        elif entity_type == "diagnosis":
            prompt   = _build_diagnosis_prompt(entity_text, current_note, prior_notes)
            response = _ollama_generate(prompt, self.base_url, self.model)
            return _parse_diagnosis_response(response, entity_text)

        return None
