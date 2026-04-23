"""
test_preflight.py — Preflight (EHR Temporal Validator)
Apoorva Kolhatkar | MHI, University of Michigan

Unit tests for core pipeline components.

Run:
    python -m pytest tests/test_preflight.py -v
    # or from project root:
    python -m pytest -v

Coverage:
    - Temporal validator: complication-before-procedure ordering violation
    - ICD divergence: condition in free text, absent from coded record
    - Trust score: boundary conditions (zero penalties → near 1.0, max penalties → floor)
    - Confidence scorer: reference date anchoring (latest note date, not today)
    - OllamaResolver: graceful degradation when Ollama unavailable

These tests run without any external dependencies (no GLiNER, no Ollama, no network).
All pipeline components under test use their rule-based / heuristic paths only.
"""

import sys
import os
import pytest

# Add src/ to path so modules are importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))


# ── Test 1: Temporal validator — complication before procedure ────────────────

class TestTemporalValidator:
    """
    The temporal validator checks that clinical events appear in a defensible
    chronological order. The core violation: a complication documented before
    the procedure that would cause it.
    """

    def _make_extracted(self, notes_with_entities):
        """Helper — build extracted entity structure from simple spec."""
        return [
            {
                "note_id": note_id,
                "date": date,
                "entities": {
                    "procedures": [{"text": p, "negated": False} for p in procs],
                    "complications": [{"text": c, "negated": False} for c in comps],
                    "medications": [],
                    "diagnoses": [],
                }
            }
            for note_id, date, procs, comps in notes_with_entities
        ]

    def test_complication_before_procedure_flagged(self):
        """
        Seroma appears in note_1 (Jan). Mastectomy appears in note_2 (Feb).
        Temporal validator should flag this as a violation.
        """
        try:
            from temporal_validator import TemporalValidator
            from temporal_anchor import TemporalAnchor
        except ImportError:
            pytest.skip("temporal_validator not available in this environment")

        notes = [
            {"note_id": "note_1", "date": "2023-01-15", "text": "Patient presents with seroma."},
            {"note_id": "note_2", "date": "2023-02-20", "text": "Patient underwent mastectomy today."},
        ]
        extracted = self._make_extracted([
            ("note_1", "2023-01-15", [], ["seroma"]),
            ("note_2", "2023-02-20", ["mastectomy"], []),
        ])

        timeline = TemporalAnchor().build_timeline(notes)
        contradictions = TemporalValidator().validate(extracted, timeline)

        # Should detect that seroma precedes its associated procedure
        assert len(contradictions) > 0, (
            "Expected at least one temporal contradiction for complication-before-procedure"
        )

    def test_correct_ordering_not_flagged(self):
        """
        Mastectomy in note_1, seroma in note_2 — correct order.
        Should produce zero contradictions.
        """
        try:
            from temporal_validator import TemporalValidator
            from temporal_anchor import TemporalAnchor
        except ImportError:
            pytest.skip("temporal_validator not available in this environment")

        notes = [
            {"note_id": "note_1", "date": "2023-01-15", "text": "Patient underwent mastectomy today."},
            {"note_id": "note_2", "date": "2023-02-20", "text": "Patient presents with seroma post-op."},
        ]
        extracted = self._make_extracted([
            ("note_1", "2023-01-15", ["mastectomy"], []),
            ("note_2", "2023-02-20", [], ["seroma"]),
        ])

        timeline = TemporalAnchor().build_timeline(notes)
        contradictions = TemporalValidator().validate(extracted, timeline)

        # Correct ordering should not be flagged
        procedure_before_complication = [
            c for c in contradictions
            if "seroma" in str(c).lower() and "mastectomy" in str(c).lower()
        ]
        assert len(procedure_before_complication) == 0, (
            "Correct procedure-then-complication ordering should not produce a contradiction"
        )


# ── Test 2: ICD divergence — free text condition missing from coded record ────

class TestICDDivergence:
    """
    ICD divergence analysis surfaces diagnoses present in free-text notes
    but absent from the patient's coded ICD record — the most common failure
    mode that causes AI systems using structured data to miss clinically
    significant conditions.
    """

    def test_free_text_diagnosis_not_in_icd(self):
        """
        Depression appears in note text. Patient's ICD codes contain only
        E11.9 (diabetes) and I10 (hypertension). Depression should appear
        in the free_text_only bucket.
        """
        try:
            from icd_divergence import ICDDivergenceAnalyzer
        except ImportError:
            pytest.skip("icd_divergence not available in this environment")

        notes = [
            {
                "note_id": "note_1",
                "date": "2023-03-10",
                "text": "Patient reports low mood and difficulty concentrating. "
                        "Likely depression. Managing diabetes and hypertension well.",
            }
        ]
        extracted = [
            {
                "note_id": "note_1",
                "date": "2023-03-10",
                "entities": {
                    "diagnoses": [
                        {"text": "depression", "negated": False},
                        {"text": "diabetes", "negated": False},
                        {"text": "hypertension", "negated": False},
                    ],
                    "medications": [],
                    "procedures": [],
                    "complications": [],
                }
            }
        ]
        icd_codes = ["E11.9", "I10"]  # Diabetes and hypertension only

        result = ICDDivergenceAnalyzer().analyze(extracted, icd_codes, raw_notes=notes)

        free_text_only = [
            d.get("diagnosis", "").lower()
            for d in result.get("free_text_only", [])
        ]

        assert any("depression" in d for d in free_text_only), (
            f"Expected 'depression' in free_text_only diagnoses. Got: {free_text_only}"
        )

    def test_coded_diagnosis_matched_in_notes(self):
        """
        Hypertension appears in both notes and ICD codes.
        Should appear in matched bucket, not free_text_only.
        """
        try:
            from icd_divergence import ICDDivergenceAnalyzer
        except ImportError:
            pytest.skip("icd_divergence not available in this environment")

        notes = [
            {
                "note_id": "note_1",
                "date": "2023-03-10",
                "text": "Hypertension well-controlled on lisinopril.",
            }
        ]
        extracted = [
            {
                "note_id": "note_1",
                "date": "2023-03-10",
                "entities": {
                    "diagnoses": [{"text": "hypertension", "negated": False}],
                    "medications": [],
                    "procedures": [],
                    "complications": [],
                }
            }
        ]
        icd_codes = ["I10"]  # Hypertension coded

        result = ICDDivergenceAnalyzer().analyze(extracted, icd_codes, raw_notes=notes)

        free_text_only = [
            d.get("diagnosis", "").lower()
            for d in result.get("free_text_only", [])
        ]

        assert not any("hypertension" in d for d in free_text_only), (
            "Hypertension is coded — should not appear in free_text_only"
        )


# ── Test 3: Trust score boundary conditions ───────────────────────────────────

class TestTrustScore:
    """
    The trust score is a penalty-based heuristic. These tests verify boundary
    behavior — not that the weights are clinically calibrated (they aren't,
    and that's documented), but that the function behaves correctly at its
    defined boundaries.
    """

    def _compute_trust(self, scored, contradictions, icd):
        """Import and call compute_trust from services."""
        try:
            from services import compute_trust
        except ImportError:
            try:
                # Try from pipeline.py if services not available
                from pipeline import compute_trust
            except ImportError:
                pytest.skip("compute_trust not importable in this environment")
        return compute_trust(scored, contradictions, icd)

    def test_no_penalties_scores_near_one(self):
        """
        Clean chart — no low-confidence entities, no copy-forward,
        no contradictions, no ICD gaps. Score should be 1.0.
        """
        scored = [
            {"composite_score": 0.85, "copy_forward_suspicion": 0.02},
            {"composite_score": 0.91, "copy_forward_suspicion": 0.01},
        ]
        contradictions = []
        icd = {"free_text_only": [], "code_only": [], "matched": []}

        score = self._compute_trust(scored, contradictions, icd)
        assert score == 1.0, f"Expected 1.0 for clean chart, got {score}"

    def test_multiple_penalties_reduces_score(self):
        """
        Chart with low-confidence entities, copy-forward suspicion,
        contradictions, and ICD gaps. Score should be meaningfully below 1.0.
        """
        # 5 low-confidence entities → 5 × 0.05 = 0.25 penalty
        scored = [{"composite_score": 0.30, "copy_forward_suspicion": 0.70}] * 5
        # 3 contradictions → 3 × 0.08 = 0.24 penalty (capped at 0.30)
        contradictions = ["c1", "c2", "c3"]
        # 2 ICD gaps → 2 × 0.03 = 0.06 penalty
        icd = {"free_text_only": [{"diagnosis": "depression"}, {"diagnosis": "anxiety"}]}

        score = self._compute_trust(scored, contradictions, icd)
        assert score < 0.8, f"Expected score well below 0.8 for high-penalty chart, got {score}"
        assert score >= 0.0, f"Score should never go negative, got {score}"

    def test_score_never_exceeds_one(self):
        """Score is capped at 1.0 regardless of inputs."""
        score = self._compute_trust([], [], {"free_text_only": []})
        assert score <= 1.0, f"Score exceeded 1.0: {score}"

    def test_score_never_goes_negative(self):
        """Score floor is 0.0 — extreme penalties don't produce negative values."""
        # Maximum possible penalties across all components
        scored = [{"composite_score": 0.10, "copy_forward_suspicion": 0.95}] * 20
        contradictions = ["c"] * 20
        icd = {"free_text_only": [{"diagnosis": f"dx_{i}"} for i in range(20)]}

        score = self._compute_trust(scored, contradictions, icd)
        assert score >= 0.0, f"Score went negative: {score}"


# ── Test 4: Confidence scorer — reference date anchoring ─────────────────────

class TestConfidenceScorer:
    """
    The confidence scorer computes temporal decay relative to a reference date.
    Critical fix: reference date should be the latest note date in the record,
    not today's date. This ensures historical records aren't penalized simply
    for being old.
    """

    def test_reference_date_uses_latest_note(self):
        """
        _latest_note_date() should return the most recent date across all notes.
        """
        try:
            from confidence_scorer import _latest_note_date
        except ImportError:
            pytest.skip("confidence_scorer not available in this environment")

        notes = [
            {"note_id": "note_1", "date": "2023-01-15"},
            {"note_id": "note_2", "date": "2023-08-30"},
            {"note_id": "note_3", "date": "2023-05-10"},
        ]

        result = _latest_note_date(notes)
        assert result == "2023-08-30", (
            f"Expected latest note date '2023-08-30', got '{result}'"
        )

    def test_reference_date_handles_missing_dates(self):
        """
        Notes with no parseable dates should not crash _latest_note_date.
        Should fall back to today's date.
        """
        try:
            from confidence_scorer import _latest_note_date
        except ImportError:
            pytest.skip("confidence_scorer not available in this environment")

        notes = [
            {"note_id": "note_1"},  # No date key
            {"note_id": "note_2", "date": None},
            {"note_id": "note_3", "date": "not-a-date"},
        ]

        # Should not raise — fallback to today
        result = _latest_note_date(notes)
        assert result is not None
        assert len(result) == 10  # YYYY-MM-DD format


# ── Test 5: OllamaResolver graceful degradation ───────────────────────────────

class TestOllamaResolverDegradation:
    """
    When Ollama is unavailable, OllamaResolver must return the input
    extracted_entities unchanged — no crash, no silent data loss.
    This is a non-negotiable requirement for production use.

    Uses unittest.mock.patch to force _ollama_available() to return False
    regardless of whether Ollama is actually running on the test machine.
    This makes the test deterministic in all environments.
    """

    def test_unavailable_ollama_returns_input_unchanged(self):
        """
        With Ollama reported as unavailable (mocked), resolve_all() should
        return the exact input list without modification.
        """
        try:
            import ollama_resolver as _mod
            from ollama_resolver import OllamaResolver
        except ImportError:
            pytest.skip("ollama_resolver not available in this environment")

        from unittest.mock import patch

        extracted = [
            {
                "note_id": "note_1",
                "date": "2023-01-15",
                "entities": {
                    "medications": [{"text": "metformin", "negated": False}],
                    "diagnoses": [{"text": "diabetes", "negated": False}],
                }
            }
        ]
        notes = [{"note_id": "note_1", "date": "2023-01-15",
                  "text": "Patient on metformin for diabetes."}]

        # Patch _ollama_available at the module level so the constructor
        # always sees False regardless of real Ollama state on this machine
        with patch.object(_mod, "_ollama_available", return_value=False):
            resolver = OllamaResolver()
            assert resolver._enabled is False, (
                "Resolver should be disabled when _ollama_available returns False"
            )

        result = resolver.resolve_all(extracted, notes)

        # Should return input unchanged when disabled
        assert result == extracted, (
            "resolve_all() should return input unchanged when Ollama is unavailable"
        )
        assert "resolved_statuses" not in result[0]["entities"], (
            "resolved_statuses should not be added when Ollama is unavailable"
        )

    def test_enabled_flag_reflects_availability(self):
        """
        _enabled should be True when Ollama is available, False when not.
        Both states are tested via mocking.
        """
        try:
            import ollama_resolver as _mod
            from ollama_resolver import OllamaResolver
        except ImportError:
            pytest.skip("ollama_resolver not available in this environment")

        from unittest.mock import patch

        with patch.object(_mod, "_ollama_available", return_value=False):
            resolver = OllamaResolver()
            assert resolver._enabled is False

        with patch.object(_mod, "_ollama_available", return_value=True):
            resolver = OllamaResolver()
            assert resolver._enabled is True


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
