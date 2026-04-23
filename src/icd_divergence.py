"""
icd_divergence.py
Compares diagnoses extracted from free text against ICD-10 coded fields.

This addresses the core problem from Michigan Medicine research:
when a diagnosis lives only in narrative text and was never coded,
ICD-based longitudinal queries miss it entirely — even though it
may be clinically significant.

Two failure modes detected:
  1. Free-text-only: Diagnosis in notes but not in ICD codes
     → Undercoding risk, missed in downstream analytics
  2. Code-only: ICD code present but never mentioned in notes
     → Possible phantom code, coding error, or copy-forward from prior encounter
"""

import re
from typing import Optional

# Mapping of common diagnosis text patterns to ICD-10 code prefixes
# Simplified for prototype — production would use a full UMLS/SNOMED mapping
DIAGNOSIS_TO_ICD_MAP = {
    "diabetes mellitus type 2": ["E11"],
    "diabetes mellitus type 1": ["E10"],
    "diabetes": ["E10", "E11", "E13"],
    "hypertension": ["I10", "I11", "I12", "I13"],
    "heart failure": ["I50"],
    "atrial fibrillation": ["I48"],
    "chronic kidney disease": ["N18"],
    "copd": ["J44"],
    "asthma": ["J45"],
    "depression": ["F32", "F33"],
    "anxiety": ["F40", "F41"],
    "hypothyroidism": ["E03"],
    "hyperthyroidism": ["E05"],
    "obesity": ["E66"],
    "anemia": ["D50", "D51", "D52", "D53", "D55", "D64"],
    "sepsis": ["A40", "A41"],
    "pneumonia": ["J18", "J15", "J13"],
    "deep vein thrombosis": ["I82"],
    "pulmonary embolism": ["I26"],
    "stroke": ["I63", "I64"],
    "myocardial infarction": ["I21", "I22"],
    "breast cancer": ["C50"],
    "invasive ductal carcinoma": ["C50"],
    "invasive lobular carcinoma": ["C50"],
    "ductal carcinoma": ["C50"],
    "carcinoma": ["C50", "C34", "C18"],
    "lung cancer": ["C34"],
    "colon cancer": ["C18"],
    "lymphoma": ["C81", "C82", "C83", "C84", "C85"],
    "leukemia": ["C91", "C92", "C93", "C94", "C95"],
    "mastectomy": ["Z90.1"],
    "seroma": ["T79"],
    "wound infection": ["T81.4"],
    "deep vein thrombosis": ["I82"],
}

# Reverse map: ICD prefix → diagnosis terms
ICD_TO_DIAGNOSIS_MAP = {}
# Ensure carcinoma variants map back
for diag, codes in DIAGNOSIS_TO_ICD_MAP.items():
    for code in codes:
        prefix = code[:3]
        if prefix not in ICD_TO_DIAGNOSIS_MAP:
            ICD_TO_DIAGNOSIS_MAP[prefix] = []
        ICD_TO_DIAGNOSIS_MAP[prefix].append(diag)


class ICDDivergenceAnalyzer:

    def analyze(self, extracted_entities: list[dict], icd_codes: list[str], raw_notes: list[dict] = None) -> dict:
        """
        Compare free-text extracted diagnoses against structured ICD codes.

        Args:
            extracted_entities: Output from EntityExtractor
            icd_codes: List of ICD-10 codes from structured EHR fields
                       e.g. ["E11.9", "I10", "C50.912"]

        Returns:
            dict with:
              - free_text_only: diagnoses in notes but not coded
              - code_only: ICD codes with no narrative support
              - matched: diagnoses confirmed in both sources
              - divergence_score: 0-1, higher = more divergence
        """
        # Extract all non-negated diagnoses from notes (true conditions only)
        free_text_diagnoses = self._collect_free_text_diagnoses(extracted_entities)
        # Include procedure entities for CODE matching (mastectomy covers Z90)
        # but NOT as candidates for "uncoded diagnosis" - procedures have different coding
        procedure_texts = list({
            e["text"].lower().strip()
            for nd in extracted_entities
            for e in nd["entities"].get("procedures", [])
            if not e.get("negated")
        })
        # Raw note text — used ONLY for code matching, not for uncoded-diagnosis detection
        raw_note_mentions = list(procedure_texts)
        if raw_notes:
            import re as _re
            for note in raw_notes:
                text = note.get("text", "").lower()
                for key in DIAGNOSIS_TO_ICD_MAP:
                    if _re.search(r"\b" + _re.escape(key) + r"\b", text):
                        raw_note_mentions.append(key)
        raw_note_mentions = list(set(raw_note_mentions))
        # Combined for code-only checking (broader)
        all_mentions = list(set(free_text_diagnoses + raw_note_mentions))
        free_text_diagnoses = list(set(free_text_diagnoses))

        # Normalize ICD codes to 3-character prefixes
        coded_prefixes = set()
        for code in icd_codes:
            coded_prefixes.add(code[:3].upper())

        # Map free-text diagnoses to expected ICD prefixes
        free_text_only = []
        matched = []
        for diag_text in free_text_diagnoses:
            expected_prefixes = self._get_expected_codes(diag_text)
            if not expected_prefixes:
                continue  # Unknown mapping — skip

            is_coded = any(p in coded_prefixes for p in expected_prefixes)
            if is_coded:
                matched.append({
                    "diagnosis": diag_text,
                    "icd_prefixes": expected_prefixes,
                    "status": "confirmed_both_sources"
                })
            else:
                free_text_only.append({
                    "diagnosis": diag_text,
                    "expected_icd_prefixes": expected_prefixes,
                    "status": "free_text_only",
                    "risk": "undercoding — may be missed in ICD-based longitudinal queries",
                    "first_seen_note": self._first_seen_note(extracted_entities, diag_text)
                })

        # Check for codes with no narrative support
        code_only = []
        for prefix in coded_prefixes:
            expected_diagnoses = ICD_TO_DIAGNOSIS_MAP.get(prefix, [])
            if not expected_diagnoses:
                continue
            narrative_support = any(
                any(d in diag_text for d in expected_diagnoses)
                for diag_text in free_text_diagnoses
            )
            if not narrative_support:
                code_only.append({
                    "icd_prefix": prefix,
                    "expected_diagnoses": expected_diagnoses,
                    "status": "code_only",
                    "risk": "phantom code risk — no narrative documentation found"
                })

        # Divergence score
        total = len(free_text_only) + len(code_only) + len(matched)
        divergence_score = round(
            (len(free_text_only) + len(code_only)) / total, 3
        ) if total > 0 else 0.0

        return {
            "free_text_only": free_text_only,
            "code_only": code_only,
            "matched": matched,
            "divergence_score": divergence_score,
            "summary": (
                f"{len(free_text_only)} diagnoses in notes not coded | "
                f"{len(code_only)} codes with no narrative support | "
                f"{len(matched)} confirmed in both"
            )
        }

    def _collect_free_text_diagnoses(self, extracted_entities: list[dict]) -> list[str]:
        seen = set()
        diagnoses = []
        for note_data in extracted_entities:
            for entity in note_data["entities"].get("diagnoses", []):
                if entity.get("negated"):
                    continue
                text = entity["text"].lower().strip()
                if text not in seen:
                    seen.add(text)
                    diagnoses.append(text)
        return diagnoses

    def _get_expected_codes(self, diag_text: str) -> list[str]:
        """
        Find ICD prefixes for a given diagnosis text.
        Uses root-word matching and synonym expansion to avoid false positives
        where the diagnosis is clearly documented but phrasing differs from
        the mapping key (e.g. 'underwent mastectomy' vs. 'mastectomy').
        """
        import re
        text = diag_text.lower().strip()
        for key, codes in DIAGNOSIS_TO_ICD_MAP.items():
            key_l = key.lower()
            # Direct match
            if key_l in text or text in key_l:
                return [c[:3] for c in codes]
            # Root word match — check if the first significant word of the key
            # appears as a whole word in the diagnosis text
            key_words = [w for w in key_l.split() if len(w) > 4]
            if key_words and all(
                re.search(r'' + re.escape(w) + r'', text)
                for w in key_words[:2]
            ):
                return [c[:3] for c in codes]
        return []

    def _text_mentions_code(self, icd_prefix: str, free_text_diagnoses: list[str]) -> bool:
        """
        Check if there is narrative support for an ICD code.
        Uses root-word matching — 'mastectomy' in notes covers Z90 code.
        """
        import re
        expected_diags = ICD_TO_DIAGNOSIS_MAP.get(icd_prefix, [])
        if not expected_diags:
            return False
        for expected in expected_diags:
            key_words = [w for w in expected.lower().split() if len(w) > 4]
            for ft in free_text_diagnoses:
                if not key_words:
                    continue
                if all(re.search(r'\b' + re.escape(w) + r'\b', ft) for w in key_words[:2]):
                    return True
        return False

    def _first_seen_note(self, extracted_entities: list[dict], diag_text: str) -> Optional[str]:
        for note_data in extracted_entities:
            for entity in note_data["entities"].get("diagnoses", []):
                if entity["text"].lower().strip() == diag_text:
                    return note_data["note_id"]
        return None
