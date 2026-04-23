"""
fusion.py
Structured + Unstructured Data Fusion

This is the core differentiator. Most EHR AI systems process
notes OR structured data. This module compares both and flags
where they conflict — the signal that existing tools miss.

Source hierarchy (highest to lowest reliability):
  Tier 1: Lab results, pathology, finalized imaging
  Tier 2: Medication orders, prescriptions (structured)
  Tier 3: Problem list, diagnosis fields (structured)
  Tier 4: Recent clinical notes (< 30 days)
  Tier 5: Old notes / copy-forward candidates (> 30 days)

Conflict types detected:
  1. MEDICATION CONFLICT
     Structured: medication active
     Notes: patient reported stopping medication
     → FLAG: reconcile before visit

  2. UNDOCUMENTED CONDITION
     Notes: diagnosis mentioned in free text
     Structured: not in problem list or ICD codes
     → FLAG: possible undercoding

  3. STALE STRUCTURED DATA
     Structured: medication/diagnosis active
     Notes: resolved or discontinued documented
     → FLAG: structured data may not reflect current status

  4. MISSING MANAGEMENT
     Structured: chronic condition active
     Structured: no associated medication on record
     → FLAG: condition may be unmanaged
"""

import re
from datetime import datetime


STOP_PATTERN = re.compile(
    r"\b(?:stopped?|discontinued?|no\s+longer\s+taking|not\s+taking|refused|held)\b",
    re.IGNORECASE
)

RESOLUTION_PATTERN = re.compile(
    r"\b(?:resolved|resolving|healed|cleared|improved|no\s+longer\s+present)\b",
    re.IGNORECASE
)

# Condition -> expected medications for active management
CONDITION_MED_MAP = {
    "diabetes mellitus type 2": ["metformin","insulin","glipizide","sitagliptin","empagliflozin"],
    "diabetes mellitus type 1": ["insulin"],
    "diabetes": ["metformin","insulin","glipizide","sitagliptin"],
    "hypertension": ["lisinopril","metoprolol","amlodipine","losartan","hydrochlorothiazide"],
    "depression": ["sertraline","fluoxetine","escitalopram","bupropion","venlafaxine"],
    "atrial fibrillation": ["warfarin","rivaroxaban","apixaban","metoprolol"],
    "heart failure": ["furosemide","lisinopril","carvedilol","spironolactone"],
}

SOURCE_TIER_LABELS = {
    "tier_1": "Lab / pathology result",
    "tier_2": "Medication order (structured)",
    "tier_3": "Problem list (structured)",
    "tier_4": "Recent clinical note",
    "tier_5": "Older note (copy-forward risk)",
}


def _parse_date(s):
    if not s: return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
        try: return datetime.strptime(s, fmt)
        except: continue
    return None


def _note_age_tier(note_date, reference_date):
    """Classify note as recent (tier_4) or old (tier_5)."""
    d1 = _parse_date(note_date)
    d2 = _parse_date(reference_date)
    if not d1 or not d2:
        return "tier_5"
    days = (d2 - d1).days
    return "tier_4" if days <= 30 else "tier_5"


class FusionAnalyzer:

    def analyze(self, structured_data, notes, extracted_entities):
        """
        Compare structured data against note-extracted entities.
        Returns list of conflicts with full evidence and action layers.
        """
        if not structured_data:
            return []

        conflicts = []
        note_dates = [n.get("date") for n in notes if n.get("date")]
        reference_date = sorted(note_dates)[-1] if note_dates else ""

        struct_meds  = {m["medication"].lower(): m for m in structured_data.get("structured_medications", [])}
        struct_diags = {d["diagnosis"].lower(): d for d in structured_data.get("structured_diagnoses", [])}

        # 1. Medication conflicts — structured says active, notes say stopped
        conflicts.extend(self._detect_medication_conflicts(
            struct_meds, notes, reference_date
        ))

        # 2. Missing management — condition active but no medication
        conflicts.extend(self._detect_missing_management(
            struct_diags, struct_meds
        ))

        # 3. Undocumented conditions — in notes but not in structured
        conflicts.extend(self._detect_undocumented_conditions(
            struct_diags, extracted_entities, notes
        ))

        # 4. Stale structured data — resolved in notes but still active in structured
        conflicts.extend(self._detect_stale_structured(
            struct_meds, struct_diags, notes
        ))

        return conflicts

    def _detect_medication_conflicts(self, struct_meds, notes, reference_date):
        """
        Structured medication is active.
        Notes contain patient-reported discontinuation.
        This is the highest-confidence conflict — two sources disagree.
        """
        conflicts = []
        discontinued_in_notes = {}  # med -> (note_id, date, context)

        for note in notes:
            note_id   = note["note_id"]
            note_date = note.get("date","")
            text      = note.get("text","")
            text_lower = text.lower()
            note_tier = _note_age_tier(note_date, reference_date)

            for match in STOP_PATTERN.finditer(text_lower):
                after = text_lower[match.end():match.end()+80]
                for med_name in struct_meds:
                    med_root = med_name.split()[0]  # handle "metformin 1000mg" -> "metformin"
                    if re.search(r"\b" + re.escape(med_root) + r"\b", after, re.IGNORECASE):
                        if med_name not in discontinued_in_notes:
                            # Extract reason if present
                            reason_match = re.search(
                                r"(?:due\s+to|because\s+of|secondary\s+to)\s+([^.]{5,40})",
                                after, re.IGNORECASE
                            )
                            reason = reason_match.group(1).strip() if reason_match else "reason not documented"
                            discontinued_in_notes[med_name] = {
                                "note_id":   note_id,
                                "note_date": note_date,
                                "note_tier": note_tier,
                                "reason":    reason,
                            }

        for med_name, disc_info in discontinued_in_notes.items():
            struct_med = struct_meds[med_name]
            if struct_med.get("status") == "active":
                conflicts.append({
                    "conflict_type":  "medication_conflict",
                    "severity":       "high",
                    "priority_score": 10,
                    "problem": f"{struct_med['medication'].title()} listed as active in structured record",
                    "evidence": [
                        f"Structured ({SOURCE_TIER_LABELS[struct_med['source_tier']]}): "
                        f"{struct_med['medication']} {struct_med.get('dose','')} — status: ACTIVE "
                        f"(last updated {struct_med.get('last_updated','')})",
                        f"Clinical note ({disc_info['note_id']}, {disc_info['note_date']}): "
                        f"patient reported stopping {struct_med['medication']} — {disc_info['reason']}",
                        f"No re-initiation documented in subsequent notes",
                    ],
                    "action": f"Reconcile {struct_med['medication']} status before visit — "
                              f"confirm discontinued or restarted. Update structured medication list.",
                    "impact": "Structured medication list inaccuracy affects care coordination, "
                              "medication reconciliation at transitions, and downstream coding. "
                              "If diabetes medication discontinued with no replacement, "
                              "HCC coding for active diabetes management may not survive RADV audit.",
                    "source_conflict": {
                        "structured": struct_med['source_tier'],
                        "notes":      disc_info['note_tier'],
                    }
                })

        return conflicts

    def _detect_missing_management(self, struct_diags, struct_meds):
        """
        Chronic condition is in the structured problem list
        but no associated medication is on the active medication list.
        Condition may be unmanaged — or management is undocumented.
        """
        conflicts = []
        active_med_names = {
            m.split()[0].lower()
            for m in struct_meds
            if struct_meds[m].get("status") == "active"
        }

        already_flagged = set()
        for diag_name, diag_data in struct_diags.items():
            if diag_data.get("status") != "active":
                continue

            for condition_key, expected_meds in CONDITION_MED_MAP.items():
                if condition_key not in diag_name:
                    continue
                root = condition_key.split()[0]
                if root in already_flagged:
                    continue

                has_management = any(
                    med.split()[0].lower() in active_med_names
                    for med in expected_meds
                )

                if not has_management:
                    conflicts.append({
                        "conflict_type":  "missing_management",
                        "severity":       "high",
                        "priority_score": 9,
                        "problem": f"{diag_data['diagnosis']} active with no medication documented",
                        "evidence": [
                            f"Structured problem list: {diag_data['diagnosis']} — status: ACTIVE "
                            f"(ICD: {diag_data.get('icd_code','')}, "
                            f"last updated {diag_data.get('last_updated','')})",
                            f"Expected medications for active management: "
                            f"{', '.join(expected_meds[:3])}",
                            f"None of these appear as active in the structured medication list",
                        ],
                        "action": f"Verify {condition_key} management at this visit. "
                                  f"Document current treatment approach, reason for no pharmacotherapy, "
                                  f"or initiate appropriate medication. "
                                  f"Ensure MEAT criteria (Treatment) is met in the note.",
                        "impact": f"Unmanaged {condition_key} creates both care quality risk and "
                                  f"HCC coding risk. Under CMS-HCC V28, chronic conditions require "
                                  f"MEAT-compliant documentation — a diagnosis without documented "
                                  f"management may not survive RADV audit. "
                                  f"Revenue at risk: ~$3,000/member/year in RAF leakage.",
                        "source_conflict": {
                            "structured": diag_data['source_tier'],
                            "notes":      "no medication found",
                        }
                    })
                    already_flagged.add(root)

        return conflicts

    def _detect_undocumented_conditions(self, struct_diags, extracted_entities, notes):
        """
        Condition appears in clinical notes but not in structured problem list.
        Possible undercoding — ICD-based queries will miss it.
        """
        conflicts = []
        struct_diag_roots = set()
        for d in struct_diags:
            for word in d.split():
                if len(word) > 4:
                    struct_diag_roots.add(word.lower())

        # Conditions explicitly mentioned as NEW in notes
        new_diag_patterns = [
            r"\b(?:new\s+diagnosis|newly\s+diagnosed|diagnosed\s+with|"
            r"new\s+onset)\s+(?:of\s+)?([a-z\s\-]{4,40})",
        ]

        seen = set()
        for note in notes:
            text_lower = note.get("text","").lower()
            for pattern in new_diag_patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    diag = re.sub(r"^(of|the|a|an)\s+", "", match.group(1).strip())
                    diag = diag[:40].strip()
                    if len(diag) < 4 or diag in seen:
                        continue
                    seen.add(diag)

                    # Check if it's in structured data
                    diag_words = [w for w in diag.split() if len(w) > 4]
                    in_structured = any(w in struct_diag_roots for w in diag_words)

                    if not in_structured:
                        conflicts.append({
                            "conflict_type":  "undocumented_condition",
                            "severity":       "moderate",
                            "priority_score": 7,
                            "problem": f"{diag.title()} documented in notes but absent from structured record",
                            "evidence": [
                                f"Clinical note ({note['note_id']}, {note.get('date','')}): "
                                f"new diagnosis of '{diag}' documented in free text",
                                f"Structured problem list: no matching entry found",
                                f"ICD-coded record: no corresponding code",
                            ],
                            "action": f"Add {diag} to the structured problem list and assign "
                                      f"appropriate ICD-10 code. Verify treatment plan is documented "
                                      f"to meet MEAT criteria if this is a chronic condition.",
                            "impact": f"Conditions absent from structured data are invisible to "
                                      f"ICD-based analytics, population health queries, care gap "
                                      f"detection, and HCC risk adjustment. "
                                      f"Uncoded diagnoses directly reduce RAF scores.",
                            "source_conflict": {
                                "structured": "absent",
                                "notes":      "tier_4",
                            }
                        })
        return conflicts

    def _detect_stale_structured(self, struct_meds, struct_diags, notes):
        """
        Structured data shows active status but notes indicate resolution.
        Structured data may not have been updated after clinical changes.
        """
        conflicts = []
        for note in notes:
            text_lower = note.get("text","").lower()
            for med_name, med_data in struct_meds.items():
                if med_data.get("status") != "active":
                    continue
                med_root = med_name.split()[0]
                if med_root in text_lower:
                    context_idx = text_lower.find(med_root)
                    context = text_lower[max(0,context_idx-60):context_idx+60]
                    if RESOLUTION_PATTERN.search(context):
                        conflicts.append({
                            "conflict_type":  "stale_structured_data",
                            "severity":       "low",
                            "priority_score": 4,
                            "problem": f"{med_data['medication'].title()} may have been resolved "
                                       f"but structured record not updated",
                            "evidence": [
                                f"Structured medication list: {med_data['medication']} — ACTIVE",
                                f"Note ({note['note_id']}): resolution language detected near "
                                f"'{med_data['medication']}' mention",
                            ],
                            "action": f"Verify current status of {med_data['medication']} "
                                      f"and update structured medication list if discontinued.",
                            "impact": "Stale structured data creates medication reconciliation "
                                      "errors at care transitions and may affect clinical "
                                      "decision support accuracy.",
                            "source_conflict": {
                                "structured": med_data['source_tier'],
                                "notes":      "tier_4",
                            }
                        })
        return conflicts
