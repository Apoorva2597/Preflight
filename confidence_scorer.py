"""
confidence_scorer.py

Composite confidence scoring for clinical entities extracted from EHR notes.

Formula:
    Composite Score = Base Score
                      × Corroboration Multiplier
                      × Temporal Decay Factor
                      × (1 − Contradiction Penalty)

Each component is computed independently and is independently auditable.
No single component can dominate the final score — they interact
multiplicatively.

Design principles:
    - Condition-category-specific decay rates (not a single global decay)
    - Corroboration requires genuine independence — duplication blocked
    - Contradiction penalties are applied sequentially, not additively
    - Output is a scored confidence object, not a binary flag
    - Every score is traceable to its component inputs

Decay constants derived from:
    Fortin et al. (2012) multimorbidity persistence rates
    van den Bussche et al. (2011) chronic condition longitudinal stability
    Singh et al. (2013) diagnostic error types and resolution intervals
    MIMIC-III discharge diagnosis persistence (Johnson et al. 2016)

These are illustrative Phase 1 calibration values.
All constants must be validated against real-world data before
production deployment.

Author: Apoorva Kolhatkar
MHI Candidate, University of Michigan
"""

import math
import re
import yaml
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


# ── Load taxonomy ─────────────────────────────────────────────────────────────

TAXONOMY_PATH = Path(__file__).parent.parent / "config" / "condition_taxonomy.yaml"

def load_taxonomy() -> dict:
    if TAXONOMY_PATH.exists():
        with open(TAXONOMY_PATH) as f:
            return yaml.safe_load(f)
    # Fallback inline if yaml not available
    return _fallback_taxonomy()

def _fallback_taxonomy() -> dict:
    return {
        "condition_categories": {
            "acute_infectious": {"lambda": 0.35, "base_score_range": [0.45, 0.75],
                "examples": ["pneumonia", "uti", "wound infection", "cellulitis", "sepsis"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": 3},
            "acute_postsurgical": {"lambda": 0.28, "base_score_range": [0.40, 0.70],
                "examples": ["seroma", "hematoma", "wound dehiscence", "ileus"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": 6},
            "episodic": {"lambda": 0.12, "base_score_range": [0.50, 0.80],
                "examples": ["deep vein thrombosis", "pulmonary embolism", "stroke", "acute mi"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": 12},
            "chronic_stable": {"lambda": 0.04, "base_score_range": [0.55, 0.85],
                "examples": ["hypertension", "hypothyroidism", "hyperlipidemia", "obesity"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": None},
            "chronic_progressive": {"lambda": 0.06, "base_score_range": [0.60, 0.88],
                "examples": ["breast cancer", "lung cancer", "chronic kidney disease", "heart failure"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": None,
                "requires_staging": True, "flag_if_unchanged_months": 6},
            "structural_anatomic": {"lambda": 0.01, "base_score_range": [0.75, 0.95],
                "examples": ["mastectomy", "colostomy", "amputation", "transplant", "spinal fusion"],
                "suppress_copy_forward_flag": True, "max_legitimate_persistence_months": None},
            "diabetes": {"lambda": 0.04, "base_score_range": [0.55, 0.85],
                "examples": ["diabetes mellitus type 2", "diabetes mellitus type 1"],
                "suppress_copy_forward_flag": False, "max_legitimate_persistence_months": None,
                "requires_medication_anchor": True,
                "medication_anchors": ["metformin", "insulin", "glipizide", "sitagliptin"]},
        },
        "source_tiers": {
            "tier_1": {"base_score_range": [0.85, 0.95]},
            "tier_2": {"base_score_range": [0.70, 0.84]},
            "tier_3": {"base_score_range": [0.45, 0.69], "default_lower_bound": 0.45},
            "tier_4": {"base_score_range": [0.20, 0.44]},
        },
        "corroboration": {
            "tier_1_objective_evidence": 1.30,
            "tier_2_independent_confirmation": 1.15,
            "same_tier_independent_institution": 1.08,
            "repeated_same_source": 1.00,
            "shared_provenance_different_feed": 1.00,
            "cap": 1.30,
        },
        "contradiction": {
            "tier_1_directly_contradicts": 0.50,
            "tier_2_interpretive_contradicts": 0.30,
            "tier_3_different_institution": 0.15,
            "tier_4_narrative_inconsistent": 0.08,
        },
        "thresholds": {
            "confirmed": [0.80, 1.00],
            "probable": [0.60, 0.79],
            "provisional": [0.40, 0.59],
            "uncertain": [0.20, 0.39],
            "contradicted": [0.00, 0.19],
        }
    }

TAXONOMY = load_taxonomy()


# ── Condition classification ──────────────────────────────────────────────────

def classify_condition(entity_text: str) -> tuple[str, dict]:
    """
    Map an entity string to its condition category and return the full
    category config. Defaults to chronic_stable if unknown — conservative
    choice that avoids over-flagging unrecognized conditions.
    """
    text = entity_text.lower().strip()
    categories = TAXONOMY.get("condition_categories", {})

    for cat_name, cat_config in categories.items():
        examples = cat_config.get("examples", [])
        if any(ex in text or text in ex for ex in examples):
            return cat_name, cat_config

    # Default — unknown conditions treated as chronic_stable
    # (low decay, low false positive risk)
    return "chronic_stable", categories.get("chronic_stable", {})


# ── Component 1: Base reliability score ──────────────────────────────────────

def compute_base_score(
    source_tier: str = "tier_3",
    institution_has_maintenance_protocol: bool = False
) -> float:
    """
    Base reliability score from source tier.
    Institutional factors modulate within tier range.
    Unknown source defaults to lower bound of Tier III (0.45).
    """
    tiers = TAXONOMY.get("source_tiers", {})
    tier_config = tiers.get(source_tier, tiers.get("tier_3", {}))
    score_range = tier_config.get("base_score_range", [0.45, 0.69])

    low, high = score_range
    # Institutional modifier: known maintenance protocol → upper half of range
    if institution_has_maintenance_protocol:
        return low + (high - low) * 0.75
    else:
        return low + (high - low) * 0.35


# ── Component 2: Temporal decay ───────────────────────────────────────────────

def compute_temporal_decay(
    condition_category: str,
    months_since_verification: float,
    cat_config: Optional[dict] = None
) -> tuple[float, str]:
    """
    Temporal decay factor using exponential decay model.
    Returns (decay_factor, rationale_string).

    decay_factor = e^(-λ × t)
    where t = months since last verification
          λ = condition-category decay constant
    """
    if cat_config is None:
        _, cat_config = classify_condition(condition_category)

    lam = cat_config.get("lambda", 0.04)  # Default to chronic_stable rate
    decay_factor = math.exp(-lam * months_since_verification)
    half_life = round(math.log(2) / lam, 1) if lam > 0 else float("inf")

    rationale = (
        f"λ={lam} (half-life {half_life}mo); "
        f"t={months_since_verification:.1f}mo since verification; "
        f"decay={decay_factor:.3f}"
    )

    return round(decay_factor, 4), rationale


# ── Component 3: Corroboration multiplier ────────────────────────────────────

def compute_corroboration_multiplier(
    corroboration_events: list[dict]
) -> tuple[float, list[str]]:
    """
    Corroboration multiplier from independent confirming evidence.

    CRITICAL: Only genuinely independent evidence counts.
    Same-source repetition, shared-provenance feeds, and patient-reported
    confirmation without clinical documentation do NOT increase the multiplier.

    corroboration_events: list of dicts with keys:
        type: one of [tier_1_objective, tier_2_independent, same_tier_independent,
                      repeated_same_source, shared_provenance]
        description: human-readable description of the corroboration event
        institution: originating institution (used for provenance check)
        originating_encounter: encounter ID if traceable (None if unknown)

    Returns (multiplier, list of rationale strings)
    """
    config = TAXONOMY.get("corroboration", {})
    cap = config.get("cap", 1.30)
    multiplier = 1.00
    rationale = []

    type_to_value = {
        "tier_1_objective": config.get("tier_1_objective_evidence", 1.30),
        "tier_2_independent": config.get("tier_2_independent_confirmation", 1.15),
        "same_tier_independent": config.get("same_tier_independent_institution", 1.08),
        "repeated_same_source": config.get("repeated_same_source", 1.00),
        "shared_provenance": config.get("shared_provenance_different_feed", 1.00),
    }

    for event in corroboration_events:
        event_type = event.get("type", "repeated_same_source")
        value = type_to_value.get(event_type, 1.00)
        desc = event.get("description", event_type)

        if value > 1.00:
            # Apply boost but enforce cap
            new_multiplier = min(multiplier * value, cap)
            if new_multiplier > multiplier:
                rationale.append(
                    f"+corroboration ({desc}): {multiplier:.3f} → {new_multiplier:.3f}"
                )
                multiplier = new_multiplier
        else:
            rationale.append(f"no boost ({desc}): repetition or shared provenance")

    if not corroboration_events:
        rationale.append("no corroboration events — multiplier unchanged at 1.00")

    return round(multiplier, 4), rationale


# ── Component 4: Contradiction penalty ───────────────────────────────────────

def compute_contradiction_penalty(
    contradictions: list[dict]
) -> tuple[float, list[str]]:
    """
    Contradiction net multiplier from conflicting evidence.
    Penalties applied sequentially (not additively) to prevent
    composite score collapsing to zero on multiple contradictions.

    contradictions: list of dicts with keys:
        tier: one of [tier_1, tier_2, tier_3, tier_4]
        description: human-readable description of the contradiction
        source: originating note or feed

    Returns (net_multiplier, list of rationale strings)
    """
    config = TAXONOMY.get("contradiction", {})
    tier_to_penalty = {
        "tier_1": config.get("tier_1_directly_contradicts", 0.50),
        "tier_2": config.get("tier_2_interpretive_contradicts", 0.30),
        "tier_3": config.get("tier_3_different_institution", 0.15),
        "tier_4": config.get("tier_4_narrative_inconsistent", 0.08),
    }

    net_multiplier = 1.00
    rationale = []

    for contradiction in contradictions:
        tier = contradiction.get("tier", "tier_4")
        penalty = tier_to_penalty.get(tier, 0.08)
        net_mult = 1.0 - penalty
        new_multiplier = net_multiplier * net_mult
        rationale.append(
            f"contradiction ({tier}, {contradiction.get('description', '')}): "
            f"×{net_mult:.2f} → multiplier {net_multiplier:.3f} → {new_multiplier:.3f}"
        )
        net_multiplier = new_multiplier

    if not contradictions:
        rationale.append("no contradictions — penalty multiplier unchanged at 1.00")

    return round(net_multiplier, 4), rationale


# ── Copy-forward signal ───────────────────────────────────────────────────────

def compute_copy_forward_signal(
    entity_text: str,
    notes: list[dict],
    cat_config: dict
) -> tuple[float, str]:
    """
    Text-similarity-based copy-forward signal.
    Returns a suspicion score (0-1) and rationale.

    For conditions where suppress_copy_forward_flag is True
    (structural/anatomic), returns 0.0 — persistence is expected.

    For chronic stable conditions, text similarity alone is insufficient.
    Medication consistency check is applied if requires_medication_anchor.
    """
    if cat_config.get("suppress_copy_forward_flag", False):
        return 0.0, "copy-forward suppressed — structural/anatomic condition"

    if len(notes) < 2:
        return 0.0, "insufficient notes for copy-forward detection"

    # Find sentences containing this entity across notes
    entity_sentences = []
    for note in notes:
        text = note.get("text", "").lower()
        if entity_text.lower() in text:
            # Extract surrounding context
            idx = text.find(entity_text.lower())
            start = max(0, idx - 100)
            end = min(len(text), idx + 200)
            entity_sentences.append((note["note_id"], note.get("date"), text[start:end]))

    if len(entity_sentences) < 2:
        return 0.0, "entity appears in fewer than 2 notes"

    # Compute pairwise similarity for consecutive appearances
    similarities = []
    for i in range(1, len(entity_sentences)):
        prev_note_id, prev_date, prev_ctx = entity_sentences[i-1]
        curr_note_id, curr_date, curr_ctx = entity_sentences[i]
        sim = SequenceMatcher(None, prev_ctx, curr_ctx).ratio()
        similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)

    # Chronic stable — high similarity is expected, not suspicious
    # Weight the suspicion score by condition type
    lam = cat_config.get("lambda", 0.04)

    # Higher lambda (acute) → similarity more suspicious
    # Lower lambda (chronic) → similarity less suspicious
    # Scale factor maps lambda range [0.01, 0.50] to suspicion weight [0.1, 1.0]
    suspicion_weight = min(1.0, max(0.1, lam / 0.35))

    raw_suspicion = max(0.0, avg_sim - 0.70) / 0.30  # Scaled to [0,1] above 0.70 threshold
    weighted_suspicion = round(raw_suspicion * suspicion_weight, 3)

    rationale = (
        f"avg similarity across {len(similarities)} consecutive appearances: {avg_sim:.3f}; "
        f"max: {max_sim:.3f}; "
        f"suspicion weight for condition type (λ={lam}): {suspicion_weight:.2f}; "
        f"copy-forward suspicion score: {weighted_suspicion:.3f}"
    )

    return weighted_suspicion, rationale


# ── Threshold classification ──────────────────────────────────────────────────

def classify_score(composite_score: float) -> str:
    thresholds = TAXONOMY.get("thresholds", {})
    for label, (low, high) in thresholds.items():
        if low <= composite_score <= high:
            return label
    return "uncertain"


def get_recommendation(
    label: str,
    condition_category: str,
    copy_forward_suspicion: float,
    months_since_verification: float,
    cat_config: dict
) -> str:
    max_persistence = cat_config.get("max_legitimate_persistence_months")
    requires_staging = cat_config.get("requires_staging", False)
    flag_if_unchanged = cat_config.get("flag_if_unchanged_months")

    parts = []

    if label == "confirmed":
        parts.append("High confidence — routine monitoring.")
    elif label == "probable":
        parts.append("Probable — passive monitoring recommended.")
    elif label == "provisional":
        parts.append("Provisional — active review recommended.")
    elif label == "uncertain":
        parts.append("Uncertain — reconciliation review triggered.")
    elif label == "contradicted":
        parts.append("CONTRADICTED — human review mandatory before clinical reliance.")

    if max_persistence and months_since_verification > max_persistence:
        parts.append(
            f"Persistence ({months_since_verification:.0f}mo) exceeds expected maximum "
            f"({max_persistence}mo) for {condition_category} — verify still active."
        )

    if copy_forward_suspicion > 0.60:
        parts.append(
            f"High copy-forward suspicion ({copy_forward_suspicion:.2f}) — "
            f"near-identical text across consecutive notes."
        )
    elif copy_forward_suspicion > 0.30:
        parts.append(
            f"Moderate copy-forward suspicion ({copy_forward_suspicion:.2f}) — "
            f"clinical updating signal weak."
        )

    if requires_staging and flag_if_unchanged and months_since_verification > flag_if_unchanged:
        parts.append(
            f"Chronic progressive condition unchanged for {months_since_verification:.0f}mo "
            f"— staging review recommended."
        )

    return " ".join(parts) if parts else "No action required."


# ── Main scoring function ─────────────────────────────────────────────────────

def score_entity(
    entity_text: str,
    notes: list[dict],
    first_seen_date: Optional[str],
    last_verified_date: Optional[str],
    source_tier: str = "tier_3",
    corroboration_events: Optional[list[dict]] = None,
    contradictions: Optional[list[dict]] = None,
    institution_has_maintenance_protocol: bool = False,
) -> dict:
    """
    Compute composite confidence score for a single clinical entity.

    Returns a fully auditable score object with component breakdown,
    classification, copy-forward suspicion, and recommendation.
    """
    corroboration_events = corroboration_events or []
    contradictions = contradictions or []

    # Classify condition
    condition_category, cat_config = classify_condition(entity_text)

    # Months since last verification
    months_since = _months_between(last_verified_date, datetime.now().strftime("%Y-%m-%d"))

    # Component 1: Base score
    base_score = compute_base_score(source_tier, institution_has_maintenance_protocol)

    # Component 2: Temporal decay
    decay_factor, decay_rationale = compute_temporal_decay(
        condition_category, months_since, cat_config
    )

    # Component 3: Corroboration
    corroboration_mult, corroboration_rationale = compute_corroboration_multiplier(
        corroboration_events
    )

    # Component 4: Contradiction penalty
    contradiction_mult, contradiction_rationale = compute_contradiction_penalty(
        contradictions
    )

    # Composite score
    composite = base_score * corroboration_mult * decay_factor * contradiction_mult
    composite = round(min(1.0, max(0.0, composite)), 4)

    # Copy-forward signal (independent of composite score)
    cf_suspicion, cf_rationale = compute_copy_forward_signal(entity_text, notes, cat_config)

    # Classification and recommendation
    label = classify_score(composite)
    recommendation = get_recommendation(
        label, condition_category, cf_suspicion, months_since, cat_config
    )

    return {
        "entity": entity_text,
        "condition_category": condition_category,
        "composite_score": composite,
        "classification": label,
        "copy_forward_suspicion": cf_suspicion,
        "recommendation": recommendation,
        "audit_trail": {
            "base_score": base_score,
            "source_tier": source_tier,
            "temporal_decay_factor": decay_factor,
            "temporal_decay_rationale": decay_rationale,
            "corroboration_multiplier": corroboration_mult,
            "corroboration_rationale": corroboration_rationale,
            "contradiction_multiplier": contradiction_mult,
            "contradiction_rationale": contradiction_rationale,
            "copy_forward_rationale": cf_rationale,
            "months_since_verification": round(months_since, 1),
            "first_seen_date": first_seen_date,
            "last_verified_date": last_verified_date,
            "formula": "composite = base × corroboration × temporal_decay × (1 − contradiction_penalty)",
        }
    }


def score_all_entities(
    extracted_entities: list[dict],
    notes: list[dict],
) -> list[dict]:
    """
    Score all extracted entities for a patient.
    Returns list of scored entity objects, sorted by composite score ascending
    (lowest confidence first — highest review priority).
    """
    scored = []

    for note_data in extracted_entities:
        note_id = note_data["note_id"]
        note_date = note_data.get("date")
        entities = note_data.get("entities", {})

        for entity_type, entity_list in entities.items():
            if entity_type == "transformer_entities":
                continue
            for entity in entity_list:
                if entity.get("negated"):
                    continue

                result = score_entity(
                    entity_text=entity["text"],
                    notes=notes,
                    first_seen_date=note_date,
                    last_verified_date=note_date,
                    source_tier="tier_3",  # Default — problem list / structured entry
                )
                result["note_id"] = note_id
                result["note_date"] = note_date
                result["entity_type"] = entity_type
                scored.append(result)

    # Sort by composite score ascending — lowest confidence = highest review priority
    scored.sort(key=lambda x: x["composite_score"])
    return scored


# ── Utilities ─────────────────────────────────────────────────────────────────

def _months_between(date_str: Optional[str], reference_str: str) -> float:
    if not date_str:
        return 6.0  # Conservative default — 6 months if unknown
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"]:
        try:
            d1 = datetime.strptime(date_str, fmt)
            d2 = datetime.strptime(reference_str, fmt)
            delta = d2 - d1
            return max(0.0, delta.days / 30.44)
        except ValueError:
            continue
    return 6.0
