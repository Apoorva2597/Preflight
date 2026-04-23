"""
EHR Temporal Validator — Main Pipeline
Apoorva Kolhatkar | Michigan Medicine NLP Research

Usage:
  python pipeline.py --input data/synthetic/expanded_patient_notes.json --output outputs/
  python pipeline.py --input data/synthetic/expanded_patient_notes.json --output outputs/ --compare
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from temporal_anchor import TemporalAnchor
from entity_extractor import EntityExtractor
from confidence_scorer import score_all_entities
from temporal_validator import TemporalValidator
from icd_divergence import ICDDivergenceAnalyzer
from named_flags import NamedFlagDetector
from timeline_output import TimelineOutput
from compare import generate_compare_html
from clinical_output import generate_clinical_html
from freshness import FreshnessAnalyzer
from care_gaps import CareGapDetector
from fusion import FusionAnalyzer
from top3_engine import select_top3
from ollama_resolver import OllamaResolver
from longitudinal_state_builder import LongitudinalStateBuilder


def load_notes(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_structured(path: str) -> list[dict]:
    """Load structured data (diagnoses, medications) if available."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def compute_trust(scored: list, contradictions: list, icd: dict) -> float:
    """
    Composite trust score — penalty-based heuristic.
    Penalties applied only to direct clinical signals.
    See services.py compute_trust for full methodology documentation.

    Penalty components:
      - High copy-forward suspicion (> 0.60):  -0.08 each, cap 0.35
      - Temporal contradictions:               -0.10 each, cap 0.40
      - Free-text-only ICD gaps:               -0.05 each, cap 0.20

    Low composite score penalty intentionally removed — see services.py.
    """
    penalty = 0.0
    if scored:
        penalty += min(
            len([e for e in scored if e.get("copy_forward_suspicion", 0) > 0.60]) * 0.08,
            0.35
        )
    penalty += min(len(contradictions) * 0.10, 0.40)
    penalty += min(len(icd.get("free_text_only", [])) * 0.05, 0.20)
    return round(max(0.0, 1.0 - penalty), 3)


def build_naive_view(extracted: list[dict]) -> dict:
    """
    Naive view: collect all non-negated entities across all notes,
    treat them all as equally reliable — no scoring, no validation.
    This is the 'trust-everything' baseline for compare mode.
    """
    meds, diags = set(), set()
    for note_data in extracted:
        for med in note_data["entities"].get("medications", []):
            if not med.get("negated"):
                meds.add(med["text"])
        for diag in note_data["entities"].get("diagnoses", []):
            if not diag.get("negated"):
                diags.add(diag["text"])
    return {
        "active_medications": sorted(meds),
        "active_diagnoses":   sorted(diags),
        "assumed_icd_codes":  [],
        "note": "Naive baseline: all entities accepted without validation."
    }


def run_pipeline(
    patient: dict,
    output_dir: str,
    extractor: EntityExtractor,
    resolver: OllamaResolver,
    compare: bool = False
) -> dict:
    pid   = patient["patient_id"]
    notes = patient["notes"]
    print(f"\n[{pid}] Processing {len(notes)} notes...")

    # Stage 1: Temporal anchoring
    timeline = TemporalAnchor().build_timeline(notes)

    # Stage 2: Entity extraction (GLiNER → BioClinicalBERT → rule-based)
    # Reuses the already-loaded model — no re-download, no re-init
    extracted = extractor.extract_all(notes)

    # Stage 2a: Longitudinal state builder
    structured_data = patient.get('structured_data')
    longitudinal_profiles = LongitudinalStateBuilder().build(
        extracted, notes, structured_data
    )

    # Stage 2b: LLM status resolution (Ollama — gracefully skipped if unavailable)
    # Reuses the already-connected resolver instance
    extracted = resolver.resolve_all(
        extracted, notes,
        candidate_profiles=longitudinal_profiles
    )

    # Stage 3: Composite confidence scoring
    scored = score_all_entities(extracted, notes)

    # Stage 4: Temporal validation
    contradictions = TemporalValidator().validate(extracted, timeline)

    # Stage 5: ICD divergence
    icd_gaps = ICDDivergenceAnalyzer().analyze(
        extracted, patient.get("icd_codes", []), raw_notes=notes
    )

    # Stage 6: Freshness analysis
    freshness = FreshnessAnalyzer().analyze(notes)

    # Stage 7: Care gap detection
    care_gaps = CareGapDetector().detect(notes, extracted)

    # Stage 8: Named flags
    named_flags = NamedFlagDetector().detect(notes, extracted)

    # Stage 9: Structured + unstructured fusion
    fusion_conflicts = FusionAnalyzer().analyze(
        patient.get("structured_data"), notes, extracted
    )

    # Stage 10: Top 3 issues
    top3 = select_top3(
        fusion_conflicts, named_flags, care_gaps,
        contradictions, icd_gaps
    )

    result = {
        "patient_id":              pid,
        "timeline":                timeline,
        "scored_entities":         scored,
        "temporal_contradictions": contradictions,
        "icd_divergence":          icd_gaps,
        "named_flags":             named_flags,
        "trust_score":             compute_trust(scored, contradictions, icd_gaps),
        "freshness":               freshness,
        "care_gaps":               care_gaps,
        "fusion_conflicts":        fusion_conflicts,
        "top3_issues":             top3,
        "longitudinal_profiles":   longitudinal_profiles,
        "resolved_statuses":       [
            s for note_data in extracted
            for s in note_data.get("entities", {}).get("resolved_statuses", [])
        ],
    }

    # Save main outputs
    out = TimelineOutput()
    out.save_json(result, os.path.join(output_dir, f"{pid}_result.json"))
    out.save_html(result, os.path.join(output_dir, f"{pid}_timeline.html"))

    # Clinical summary view
    generate_clinical_html(
        result,
        os.path.join(output_dir, f"{pid}_clinical.html")
    )

    # Compare mode
    if compare:
        naive = build_naive_view(extracted)
        generate_compare_html(
            naive, result,
            os.path.join(output_dir, f"{pid}_compare.html")
        )

    return result


def run_eval(results: list[dict]) -> dict:
    return {
        "n_patients": len(results),
        "total_named_flags": sum(len(r.get("named_flags", [])) for r in results),
        "total_high_cf_entities": sum(
            len([e for e in r.get("scored_entities", [])
                 if e.get("copy_forward_suspicion", 0) > 0.60])
            for r in results
        ),
        "total_low_confidence_entities": sum(
            len([e for e in r.get("scored_entities", [])
                 if e.get("composite_score", 1) < 0.40])
            for r in results
        ),
        "total_temporal_contradictions": sum(
            len(r["temporal_contradictions"]) for r in results
        ),
        "total_uncoded_diagnoses": sum(
            len(r["icd_divergence"].get("free_text_only", [])) for r in results
        ),
        "average_trust_score": round(
            sum(r["trust_score"] for r in results) / len(results), 3
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="EHR Temporal Validator Pipeline")
    parser.add_argument("--input",   required=True, help="Path to patient notes JSON")
    parser.add_argument("--output",  default="outputs/", help="Output directory")
    parser.add_argument("--compare", action="store_true",
                        help="Generate side-by-side naive vs. validated comparison HTML")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading notes...")
    patients = load_notes(args.input)
    print(f"Loaded {len(patients)} patients.")

    # Load heavy models once — shared across all patients
    print("Initializing models...")
    extractor = EntityExtractor()
    resolver  = OllamaResolver()

    results = [
        run_pipeline(p, args.output, extractor, resolver, compare=args.compare)
        for p in patients
    ]
    summary = run_eval(results)

    with open(os.path.join(args.output, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Eval Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs saved to {args.output}")
    if args.compare:
        print("Compare reports: open *_compare.html in your browser")


if __name__ == "__main__":
    main()
