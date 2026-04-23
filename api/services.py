"""
Preflight — EHR Temporal Validator — Service Layer
Wraps the existing pipeline for use by the API layer.
No pipeline logic lives here — this is purely an adapter.

KEY CHANGE: EntityExtractor and OllamaResolver are no longer instantiated
here. They are initialized once at API startup (main.py lifespan) and
retrieved from app.state via get_pipeline_resources(). This eliminates
the ~30–60 s GLiNER reload that previously occurred on every request.
"""
import sys
import os
import time
from typing import Optional

from fastapi import Request

# Add src/ to path so pipeline modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal_anchor import TemporalAnchor
from entity_extractor import EntityExtractor
from ollama_resolver import OllamaResolver
from confidence_scorer import score_all_entities
from temporal_validator import TemporalValidator
from icd_divergence import ICDDivergenceAnalyzer
from named_flags import NamedFlagDetector
from freshness import FreshnessAnalyzer
from care_gaps import CareGapDetector
from fusion import FusionAnalyzer
from top3_engine import select_top3

from .schemas import ValidateRequest, ValidateResponse, Top3Issue, ICDDivergenceSummary


# ── Shared resource accessor ──────────────────────────────────────────────────

def get_pipeline_resources(request: Request) -> dict:
    """
    Retrieve shared, pre-initialized pipeline resources from app.state.

    These are initialized once at startup in main.py's lifespan handler.
    Raises RuntimeError if accessed before startup completes (shouldn't
    happen in normal operation, but protects against test harness misuse).
    """
    state = request.app.state
    if not hasattr(state, "entity_extractor") or not hasattr(state, "ollama_resolver"):
        raise RuntimeError(
            "Pipeline resources not initialized. "
            "Ensure the FastAPI lifespan handler in main.py has run."
        )
    return {
        "entity_extractor": state.entity_extractor,
        "ollama_resolver": state.ollama_resolver,
    }


# ── Trust score ───────────────────────────────────────────────────────────────

def compute_trust(scored: list, contradictions: list, icd: dict) -> float:
    """
    Composite trust score — penalty-based heuristic.

    METHODOLOGY NOTE: Weights are calibrated to the synthetic ground truth
    cohort. They are not derived from clinical outcomes data. The score is a
    relative signal within a cohort, not an absolute clinical accuracy claim.
    Real-world calibration against clinician-annotated charts is the defined
    next validation step.

    Penalty components:
      - High copy-forward suspicion (> 0.60):  –0.08 each, cap 0.35
      - Temporal contradictions:               –0.10 each, cap 0.40
      - Free-text-only ICD gaps:               –0.05 each, cap 0.20

    DESIGN DECISION — low composite score penalty removed:
      Earlier versions penalised entities with composite scores < 0.40.
      This was removed because composite scores in the 0.33–0.45 range are
      structurally expected given tier_3 base scores (range 0.45–0.69) and
      temporal decay — they reflect NER extraction confidence and note age,
      not clinical reliability signals. Penalising them caused clean records
      with many notes (long longitudinal histories) to score lower than records
      with fewer notes, which is the wrong clinical signal.

      The three retained penalty components are all direct clinical findings
      that can be named, explained, and defended independently:
        - Copy-forward suspicion: near-identical entity text across notes
          after a documented status change
        - Temporal contradictions: clinical event ordering violations
        - ICD divergence: diagnosis in notes, absent from coded record
    """
    penalty = 0.0
    if scored:
        # High copy-forward suspicion — direct documentation error signal
        penalty += min(
            len([e for e in scored if e.get("copy_forward_suspicion", 0) > 0.60]) * 0.08,
            0.35
        )
    # Temporal contradictions — direct clinical ordering violation
    penalty += min(len(contradictions) * 0.10, 0.40)
    # ICD divergence — diagnosis invisible to downstream structured systems
    penalty += min(len(icd.get("free_text_only", [])) * 0.05, 0.20)
    return round(max(0.0, 1.0 - penalty), 3)


# ── Main pipeline runner ──────────────────────────────────────────────────────

def run_validation(
    request: ValidateRequest,
    entity_extractor: Optional[EntityExtractor] = None,
    ollama_resolver: Optional[OllamaResolver] = None,
) -> ValidateResponse:
    """
    Runs the full validation pipeline on a patient record.

    Accepts pre-initialized entity_extractor and ollama_resolver so that
    the API layer can pass in the shared app.state instances (no reload).
    If called directly (e.g. from CLI or tests) without passing instances,
    falls back to creating local instances — preserving backwards compat
    with pipeline.py and test scripts.
    """
    start = time.time()

    # SCALE NOTE: Pipeline processes one patient record sequentially by design.
    # This is intentional for prototype clarity and auditability.
    # Production path for high-throughput systems (50k+ encounters/night):
    #   - Async execution per patient (FastAPI BackgroundTasks or Celery)
    #   - Parallelized entity extraction across patients
    #   - GLiNER is the bottleneck — GPU inference or pre-warmed model pool
    #     would be the first optimization target
    #   - LLM resolution (Stage 2b) is already pre-filtered to unstable entities
    #     only — the longitudinal state builder is the key architectural decision
    #     that makes this viable at scale

    # Fall back to local instantiation if not injected
    # (This path is only hit from CLI / tests, not from the API)
    if entity_extractor is None:
        entity_extractor = EntityExtractor()
    if ollama_resolver is None:
        ollama_resolver = OllamaResolver()

    # Convert Pydantic models to plain dicts for pipeline compatibility
    notes = [n.dict() for n in request.notes]
    patient = {
        "patient_id": request.patient_id,
        "notes": notes,
        "icd_codes": request.icd_codes or [],
        "structured_data": request.structured_data.dict() if request.structured_data else None,
    }

    # Stage 1: Temporal anchoring
    timeline = TemporalAnchor().build_timeline(notes)

    # Stage 2: Entity extraction — uses shared GLiNER instance
    extracted = entity_extractor.extract_all(notes)

    # Stage 2b: Ollama status resolution — uses shared Ollama instance
    extracted = ollama_resolver.resolve_all(extracted, notes)

    # Stage 3: Composite confidence scoring
    scored = score_all_entities(extracted, notes)

    # Stage 4: Temporal validation
    contradictions = TemporalValidator().validate(extracted, timeline)

    # Stage 5: ICD divergence
    icd_gaps = ICDDivergenceAnalyzer().analyze(
        extracted, patient["icd_codes"], raw_notes=notes
    )

    # Stage 6: Freshness analysis
    freshness = FreshnessAnalyzer().analyze(notes)

    # Stage 7: Care gap detection
    care_gaps = CareGapDetector().detect(notes, extracted)

    # Stage 8: Named flags
    named_flags = NamedFlagDetector().detect(notes, extracted)

    # Stage 9: Fusion analysis
    fusion_conflicts = FusionAnalyzer().analyze(
        patient.get("structured_data"), notes, extracted
    )

    # Stage 10: Top 3 issues
    top3_raw = select_top3(
        fusion_conflicts, named_flags, care_gaps,
        contradictions, icd_gaps
    )

    # Trust score
    trust = compute_trust(scored, contradictions, icd_gaps)

    # Format top3 for response
    top3_issues = [
        Top3Issue(
            rank=idx + 1,
            conflict_type=i.get("conflict_type", "unknown"),
            severity=i.get("severity", "medium"),
            problem=i.get("problem", ""),
            evidence=i.get("evidence", []),
            action=i.get("action", "Review chart"),
            impact=i.get("impact"),
            confidence=i.get("confidence", "Medium"),
            source=i.get("source"),
        )
        for idx, i in enumerate(top3_raw)
    ]

    # Format ICD divergence for response
    icd_summary = ICDDivergenceSummary(
        summary=icd_gaps.get("summary", ""),
        free_text_only=[
            d.get("diagnosis", "") for d in icd_gaps.get("free_text_only", [])
        ],
        code_only=[
            d.get("icd_prefix", "") for d in icd_gaps.get("code_only", [])
        ],
        matched=[
            d.get("diagnosis", "") for d in icd_gaps.get("matched", [])
        ],
    )

    # Freshness — may be None if insufficient notes
    freshness_score = freshness.get("overall_freshness")
    if freshness_score is not None:
        freshness_score = round(freshness_score, 3)

    elapsed_ms = int((time.time() - start) * 1000)

    return ValidateResponse(
        patient_id=request.patient_id,
        trust_score=trust,
        record_freshness=freshness_score,
        top3_issues=top3_issues,
        icd_divergence=icd_summary,
        temporal_contradictions=contradictions,
        named_flags=named_flags,
        processing_time_ms=elapsed_ms,
    )
