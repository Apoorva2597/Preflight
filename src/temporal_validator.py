"""
temporal_validator.py
Validates temporal consistency of clinical events.

Core logic from Michigan Medicine surgical staging work:
  - Complications should follow their associated procedure
  - Treatments should follow their associated diagnosis
  - Active medications should not precede their indication
  - A resolved condition should not reappear as newly diagnosed

This is the rule-based temporal validation layer developed for
extracting post-surgical complication timelines in breast reconstruction research.
"""

import re
from datetime import datetime
from typing import Optional


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# Procedure → expected complication mapping
# Based on surgical staging logic from MichMed NLP pipeline
PROCEDURE_COMPLICATION_MAP = {
    "mastectomy": ["seroma", "hematoma", "infection", "lymphedema", "wound dehiscence"],
    "lumpectomy": ["seroma", "hematoma", "infection", "wound dehiscence"],
    "reconstruction": ["implant failure", "capsular contracture", "skin necrosis", "seroma", "infection"],
    "tissue expander": ["implant failure", "infection", "skin necrosis"],
    "diep flap": ["skin necrosis", "wound dehiscence", "flap failure", "seroma"],
    "tram flap": ["skin necrosis", "wound dehiscence", "flap failure", "hernia"],
    "axillary dissection": ["lymphedema", "seroma", "nerve injury"],
    "cholecystectomy": ["bile leak", "infection", "ileus"],
    "appendectomy": ["wound infection", "abscess", "ileus"],
    "colectomy": ["anastomotic leak", "ileus", "wound infection", "abscess"],
    "hip replacement": ["deep vein thrombosis", "pulmonary embolism", "infection", "dislocation"],
    "knee replacement": ["deep vein thrombosis", "pulmonary embolism", "infection", "stiffness"],
}

# Diagnosis → expected treatment mapping
DIAGNOSIS_TREATMENT_MAP = {
    "diabetes": ["metformin", "insulin", "glipizide", "sitagliptin"],
    "hypertension": ["lisinopril", "metoprolol", "amlodipine", "losartan", "hydrochlorothiazide"],
    "infection": ["vancomycin", "ceftriaxone", "piperacillin", "antibiotics", "wound care"],
    "deep vein thrombosis": ["heparin", "warfarin", "enoxaparin", "rivaroxaban", "apixaban"],
    "pulmonary embolism": ["heparin", "warfarin", "enoxaparin", "rivaroxaban", "apixaban"],
    "pain": ["morphine", "oxycodone", "hydromorphone", "acetaminophen", "ibuprofen", "gabapentin"],
    "depression": ["sertraline", "fluoxetine", "escitalopram", "bupropion", "venlafaxine"],
    "breast cancer": ["tamoxifen", "letrozole", "anastrozole", "exemestane", "chemotherapy", "radiation"],
}


class TemporalValidator:

    def validate(self, extracted_entities: list[dict], timeline: list[dict]) -> list[dict]:
        """
        Check temporal ordering of clinical events.
        Returns list of detected contradictions.
        """
        contradictions = []

        # Build indexed lookups
        note_dates = {
            e["note_id"]: parse_date(e["date"])
            for e in extracted_entities
        }
        procedure_first_seen = self._first_seen(extracted_entities, "procedures")
        complication_first_seen = self._first_seen(extracted_entities, "complications")
        diagnosis_first_seen = self._first_seen(extracted_entities, "diagnoses")
        medication_first_seen = self._first_seen(extracted_entities, "medications")

        # 1. Complication before procedure
        contradictions.extend(
            self._check_complication_before_procedure(
                procedure_first_seen, complication_first_seen, note_dates
            )
        )

        # 2. Orphaned complications (complication with no matching procedure)
        contradictions.extend(
            self._check_orphaned_complications(
                procedure_first_seen, complication_first_seen
            )
        )

        # 3. Treatment before diagnosis
        contradictions.extend(
            self._check_treatment_before_diagnosis(
                diagnosis_first_seen, medication_first_seen, note_dates
            )
        )

        # 4. Resolved diagnosis reappears as new
        contradictions.extend(
            self._check_resolved_reappearance(extracted_entities)
        )

        return contradictions

    def _first_seen(self, entities: list[dict], entity_type: str) -> dict:
        """Return {entity_text_lower: (note_id, date)} for first occurrence of each entity."""
        first = {}
        for note_data in entities:
            note_id = note_data["note_id"]
            date = note_data.get("date")
            for entity in note_data["entities"].get(entity_type, []):
                if entity.get("negated"):
                    continue
                key = entity["text"].lower().strip()
                if key not in first:
                    first[key] = (note_id, date)
        return first

    def _check_complication_before_procedure(
        self,
        procedures: dict,
        complications: dict,
        note_dates: dict
    ) -> list[dict]:
        flags = []
        for proc_text, proc_variants in PROCEDURE_COMPLICATION_MAP.items():
            # Find if this procedure was documented
            proc_note = None
            for p_key, (p_note_id, p_date) in procedures.items():
                if proc_text in p_key:
                    proc_note = (p_note_id, p_date)
                    break

            if not proc_note:
                continue

            proc_date = parse_date(proc_note[1])

            for comp_text in proc_variants:
                for c_key, (c_note_id, c_date) in complications.items():
                    if comp_text in c_key:
                        comp_date = parse_date(c_date)
                        if comp_date and proc_date and comp_date < proc_date:
                            flags.append({
                                "contradiction_type": "complication_before_procedure",
                                "complication": c_key,
                                "complication_note": c_note_id,
                                "complication_date": c_date,
                                "procedure": proc_text,
                                "procedure_note": proc_note[0],
                                "procedure_date": proc_note[1],
                                "severity": "high",
                                "note": f"'{comp_text}' documented before '{proc_text}' — temporal inconsistency."
                            })
        return flags

    def _check_orphaned_complications(
        self,
        procedures: dict,
        complications: dict
    ) -> list[dict]:
        """Flag complications that appear without any plausibly related procedure."""
        flags = []
        documented_procedures = set(procedures.keys())

        for comp_text, (comp_note_id, comp_date) in complications.items():
            matched = False
            for proc_text, expected_complications in PROCEDURE_COMPLICATION_MAP.items():
                if any(exp in comp_text for exp in expected_complications):
                    if any(proc_text in dp for dp in documented_procedures):
                        matched = True
                        break
            if not matched:
                # Only flag surgical complications without a procedure — not general ones
                surgical_complications = ["seroma", "hematoma", "skin necrosis", "flap failure",
                                          "dehiscence", "capsular contracture", "implant failure"]
                if any(sc in comp_text for sc in surgical_complications):
                    flags.append({
                        "contradiction_type": "orphaned_surgical_complication",
                        "complication": comp_text,
                        "complication_note": comp_note_id,
                        "complication_date": comp_date,
                        "severity": "moderate",
                        "note": f"Surgical complication '{comp_text}' documented with no matching procedure in record."
                    })
        return flags

    def _check_treatment_before_diagnosis(
        self,
        diagnoses: dict,
        medications: dict,
        note_dates: dict
    ) -> list[dict]:
        flags = []
        for diag_text, expected_treatments in DIAGNOSIS_TREATMENT_MAP.items():
            diag_entry = None
            for d_key, d_val in diagnoses.items():
                if diag_text in d_key:
                    diag_entry = d_val
                    break

            if not diag_entry:
                continue

            diag_date = parse_date(diag_entry[1])

            for treatment in expected_treatments:
                for med_key, (med_note_id, med_date) in medications.items():
                    if treatment in med_key:
                        med_dt = parse_date(med_date)
                        if med_dt and diag_date and med_dt < diag_date:
                            flags.append({
                                "contradiction_type": "treatment_before_diagnosis",
                                "treatment": med_key,
                                "treatment_note": med_note_id,
                                "treatment_date": med_date,
                                "diagnosis": diag_text,
                                "diagnosis_date": diag_entry[1],
                                "severity": "low",
                                "note": f"Treatment '{treatment}' precedes diagnosis '{diag_text}' — may indicate pre-existing condition or copy-forward."
                            })
        return flags

    def _check_resolved_reappearance(self, extracted_entities: list[dict]) -> list[dict]:
        """
        Flag diagnoses marked as 'resolved' or 'history of' that later
        reappear as active or newly diagnosed.
        """
        flags = []
        resolved_diagnoses = {}
        resolved_patterns = [r"\b(resolved|history of|h/o|hx of|prior|previous|remote)\b"]
        active_patterns = [r"\b(new|active|current|diagnosed with|diagnosis of|presenting with)\b"]

        for i, note_data in enumerate(extracted_entities):
            note_id = note_data["note_id"]
            note_date = note_data.get("date")
            text_lower = note_data.get("text", "").lower() if "text" in note_data else ""

            for diag in note_data["entities"].get("diagnoses", []):
                diag_text = diag["text"].lower()
                context_start = max(0, diag.get("start", 0) - 60)
                context = text_lower[context_start:diag.get("start", 0) + 60]

                is_resolved = any(
                    re.search(p, context, re.IGNORECASE)
                    for p in resolved_patterns
                )
                is_active = any(
                    re.search(p, context, re.IGNORECASE)
                    for p in active_patterns
                )

                if is_resolved and not is_active:
                    resolved_diagnoses[diag_text] = (note_id, note_date)
                elif is_active and diag_text in resolved_diagnoses:
                    prev_note_id, prev_date = resolved_diagnoses[diag_text]
                    if prev_note_id != note_id:
                        flags.append({
                            "contradiction_type": "resolved_diagnosis_reappears",
                            "diagnosis": diag_text,
                            "resolved_in_note": prev_note_id,
                            "resolved_date": prev_date,
                            "reappears_in_note": note_id,
                            "reappears_date": note_date,
                            "severity": "moderate",
                            "note": f"'{diag_text}' marked resolved/historical then reappears as active — possible copy-forward or true recurrence."
                        })

        return flags
