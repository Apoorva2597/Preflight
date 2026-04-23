"""
care_gaps.py
Detects care gaps from the clinical record:

  1. New diagnosis without follow-up documentation
     — condition appears once (especially in last note) with no
       subsequent mention, no coded entry, no treatment documented

  2. Medication discontinuation without documented reason
     — stopped medication, reason absent from note

  3. Medication discontinuation without replacement
     — chronic condition medication stopped, no alternative started
       (e.g. diabetes med discontinued, no new agent documented)

  4. Active management gap
     — condition documented repeatedly but associated medication
       discontinued and no replacement documented
"""

import re

KNOWN_MEDS = [
    "metformin","lisinopril","amlodipine","sertraline","warfarin","insulin",
    "aspirin","oxycodone","cephalexin","tamoxifen","letrozole","gabapentin",
    "prednisone","vancomycin","ceftriaxone","metoprolol","atorvastatin",
    "heparin","losartan","hydrochlorothiazide","glipizide","sitagliptin",
    "fluoxetine","escitalopram","bupropion","empagliflozin","liraglutide",
]

STOP_PATTERN = re.compile(
    r"\b(?:stopped?|discontinued?|no\s+longer\s+taking|not\s+taking|refused|held)\b",
    re.IGNORECASE
)

REASON_PATTERNS = [
    r"due\s+to\s+([^.]+)",
    r"because\s+of\s+([^.]+)",
    r"secondary\s+to\s+([^.]+)",
    r"(?:side\s+effects?|intolerance|allergy|adverse|gi\s+upset|nausea|"
    r"vomiting|rash|cost|non[-\s]?compliance)\b",
]

# Condition -> medications that indicate active management
CONDITION_MED_MAP = {
    "diabetes mellitus type 2": ["metformin","insulin","glipizide","sitagliptin","empagliflozin","liraglutide"],
    "diabetes mellitus type 1": ["insulin"],
    "diabetes": ["metformin","insulin","glipizide","sitagliptin","empagliflozin"],
    "hypertension": ["lisinopril","metoprolol","amlodipine","losartan","hydrochlorothiazide"],
    "depression": ["sertraline","fluoxetine","escitalopram","bupropion","venlafaxine"],
    "breast cancer": ["tamoxifen","letrozole","anastrozole","exemestane"],
    "deep vein thrombosis": ["warfarin","heparin","enoxaparin","rivaroxaban","apixaban"],
    "atrial fibrillation": ["warfarin","rivaroxaban","apixaban","metoprolol","digoxin"],
}


def has_reason(text_after_stop, full_text_lower):
    for pattern in REASON_PATTERNS:
        if re.search(pattern, text_after_stop, re.IGNORECASE):
            return True
        if re.search(pattern, full_text_lower, re.IGNORECASE):
            return True
    return False


class CareGapDetector:

    def detect(self, notes, extracted_entities):
        gaps = []
        gaps.extend(self._new_diagnosis_no_followup(notes, extracted_entities))
        gaps.extend(self._discontinuation_no_reason(notes))
        gaps.extend(self._active_management_gap(notes, extracted_entities))
        return gaps

    def _new_diagnosis_no_followup(self, notes, extracted_entities):
        """
        A NEW diagnosis (not 'history of') that appears only once,
        in or near the last note, with no coded ICD entry and no
        subsequent mention.
        """
        gaps = []
        if len(notes) < 2:
            return gaps

        last_note_ids = {notes[-1]["note_id"], notes[-2]["note_id"] if len(notes) > 1 else ""}
        new_diag_patterns = [
            r"\b(?:new\s+diagnosis|newly\s+diagnosed|diagnosed\s+with|"
            r"diagnosis\s+of|presents?\s+with\s+new)\s+([a-z\s\-]+)",
        ]

        for note in notes:
            note_id = note["note_id"]
            text_lower = note.get("text", "").lower()

            for pattern in new_diag_patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    import re as _re
                    diag_text = match.group(1).strip()[:40]
                    # Remove leading articles/prepositions captured by regex
                    diag_text = _re.sub(r"^(of|the|a|an)\s+", "", diag_text, flags=_re.IGNORECASE).strip()
                    if len(diag_text) < 4:
                        continue

                    # Check if it appears in any subsequent note
                    subsequent_mentions = sum(
                        1 for n in notes
                        if n["note_id"] != note_id
                        and n["note_id"] > note_id
                        and diag_text[:10] in n.get("text","").lower()
                    )

                    if subsequent_mentions == 0 and note_id in last_note_ids:
                        gaps.append({
                            "gap_type":   "new_diagnosis_no_followup",
                            "severity":   "moderate",
                            "diagnosis":  diag_text,
                            "note_id":    note_id,
                            "date":       note.get("date",""),
                            "detail": (
                                f"New diagnosis of '{diag_text}' documented in {note_id} "
                                f"({note.get('date','')}) with no subsequent mention, "
                                f"no coded ICD entry, and no documented treatment plan. "
                                f"Possible documentation gap or lost-to-follow-up."
                            ),
                        })
        return gaps

    def _discontinuation_no_reason(self, notes):
        """
        Medication stopped without a documented reason.
        """
        gaps = []
        for note in notes:
            text = note.get("text", "")
            text_lower = text.lower()

            for match in STOP_PATTERN.finditer(text_lower):
                after = text_lower[match.end():match.end() + 80]
                for med in KNOWN_MEDS:
                    if re.search(r"\b" + med + r"\b", after, re.IGNORECASE):
                        reason_found = has_reason(after, text_lower)
                        if not reason_found:
                            gaps.append({
                                "gap_type":   "discontinuation_no_reason",
                                "severity":   "low",
                                "medication": med,
                                "note_id":    note["note_id"],
                                "date":       note.get("date",""),
                                "detail": (
                                    f"'{med}' discontinued in {note['note_id']} "
                                    f"({note.get('date','')}) with no documented reason. "
                                    f"Reason for discontinuation should be recorded for "
                                    f"continuity of care and future prescribing decisions."
                                ),
                            })
        return gaps

    def _active_management_gap(self, notes, extracted_entities):
        """
        Condition documented repeatedly but its primary medication
        was discontinued and no replacement was started.
        """
        gaps = []

        # Find all non-negated diagnoses across record
        all_diagnoses = set()
        for nd in extracted_entities:
            for d in nd["entities"].get("diagnoses", []):
                if not d.get("negated"):
                    all_diagnoses.add(d["text"].lower().strip())

        # Find all medications ever active (non-negated, non-discontinued)
        ever_active_meds = set()
        discontinued_meds = set()
        for note in notes:
            text_lower = note.get("text","").lower()
            for med in KNOWN_MEDS:
                if re.search(r"\b" + med + r"\b", text_lower, re.IGNORECASE):
                    ever_active_meds.add(med)
            for match in STOP_PATTERN.finditer(text_lower):
                after = text_lower[match.end():match.end()+80]
                for med in KNOWN_MEDS:
                    if re.search(r"\b" + med + r"\b", after, re.IGNORECASE):
                        discontinued_meds.add(med)

        # Check each condition — deduplicate by root word to avoid
        # flagging both "diabetes" and "diabetes mellitus type 2"
        already_flagged_roots = set()
        for condition, expected_meds in CONDITION_MED_MAP.items():
            # Skip if a more specific version already flagged
            root = condition.split()[0]
            if root in already_flagged_roots:
                continue
            # Is this condition in the record?
            condition_present = any(
                condition in diag or diag in condition
                for diag in all_diagnoses
            )
            if not condition_present:
                continue

            # Were the expected meds ever active?
            meds_ever_used = [m for m in expected_meds if m in ever_active_meds]
            if not meds_ever_used:
                continue

            # Were all of them discontinued?
            all_discontinued = all(m in discontinued_meds for m in meds_ever_used)
            # Are any still active (present in last note)?
            last_note_text = notes[-1].get("text","").lower() if notes else ""
            any_still_active = any(
                re.search(r"\b" + m + r"\b", last_note_text, re.IGNORECASE)
                for m in expected_meds
                if m not in discontinued_meds
            )

            if all_discontinued and not any_still_active:
                gaps.append({
                    "gap_type":   "active_management_gap",
                    "severity":   "high",
                    "condition":  condition,
                    "discontinued_meds": meds_ever_used,
                    "detail": (
                        f"'{condition.title()}' is documented across the record "
                        f"but {', '.join(meds_ever_used)} "
                        f"{'was' if len(meds_ever_used)==1 else 'were'} discontinued "
                        f"with no documented replacement. "
                        f"This condition may be unmanaged at the time of the last visit."
                    ),
                })
                already_flagged_roots.add(root)

        return gaps
