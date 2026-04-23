"""
top3_engine.py
Selects and ranks the top 3 most clinically significant issues
from all detected signals across the pipeline.

Ranking criteria:
  1. Severity (high > moderate > low)
  2. Priority score (set per conflict type)
  3. Clinical impact (structured conflicts > note-only signals)
  4. Actionability (issues with clear action > ambiguous ones)

Output format per issue:
  - problem: one sentence, plain English
  - evidence: list of specific supporting facts
  - action: what should happen before/during this visit
  - impact: clinical + financial consequence of ignoring it
  - source: where each piece of evidence comes from (structured vs notes)
  - confidence: plain English explanation of certainty
"""


SEVERITY_RANK = {"high": 3, "moderate": 2, "low": 1}


def _explain_confidence(conflict):
    """
    Convert technical signals into plain English confidence explanation.
    No numbers. Just reasons.
    """
    ct = conflict.get("conflict_type","")
    sc = conflict.get("source_conflict", {})
    src_struct = sc.get("structured","")
    src_notes  = sc.get("notes","")

    if ct == "medication_conflict":
        return (
            "High confidence — two independent sources disagree. "
            "Structured medication list shows active prescription; "
            "clinical note contains explicit patient-reported discontinuation. "
            "This is a direct structured-vs-narrative conflict, not an inference."
        )
    elif ct == "missing_management":
        return (
            "High confidence — structured problem list confirms condition is active; "
            "structured medication list confirms no associated medication is prescribed. "
            "Both signals come from structured data, not inference from notes."
        )
    elif ct == "undocumented_condition":
        return (
            "Moderate confidence — condition detected in clinical note free text "
            "with new-diagnosis language. Absent from structured problem list and "
            "ICD-coded record. Confidence limited by single-source documentation."
        )
    elif ct == "active_management_gap":
        return (
            "High confidence — chronic condition documented repeatedly across record; "
            "primary medication discontinued in notes with no replacement documented. "
            "Pattern consistent across multiple notes."
        )
    elif ct == "medication_copy_forward":
        return (
            "High confidence — medication listed as active after patient explicitly "
            "reported stopping it. Discontinuation note and subsequent active listing "
            "both present in record. Classic copy-forward pattern."
        )
    elif ct == "new_diagnosis_no_followup":
        return (
            "Moderate confidence — new diagnosis appears in final note with no "
            "subsequent documentation, no ICD code, and no treatment plan. "
            "May represent documentation gap or lost-to-follow-up."
        )
    elif ct == "acute_complication_persistence":
        return (
            "Moderate confidence — acute complication persists in notes beyond "
            "expected clinical resolution window without documented resolution. "
            "Temporal pattern suggests copy-forward rather than ongoing condition."
        )
    else:
        return (
            "Moderate confidence — pattern detected across multiple notes. "
            "Clinical review recommended to confirm."
        )


def _priority_score(item):
    """Compute ranking score for sorting."""
    base = item.get("priority_score", 5)
    severity_bonus = SEVERITY_RANK.get(item.get("severity","low"), 1) * 3
    # Structured conflicts rank higher than note-only
    sc = item.get("source_conflict",{})
    struct_bonus = 2 if sc.get("structured","") not in ("absent","no medication found") else 0
    return base + severity_bonus + struct_bonus


def select_top3(fusion_conflicts, named_flags, care_gaps, contradictions, icd_gaps):
    """
    Pool all detected issues, rank them, return top 3 with full action layer.
    """
    all_issues = []

    # Fusion conflicts (highest priority — structured vs notes)
    for c in fusion_conflicts:
        all_issues.append({
            "conflict_type":  c["conflict_type"],
            "severity":       c["severity"],
            "priority_score": _priority_score(c),
            "problem":        c["problem"],
            "evidence":       c["evidence"],
            "action":         c["action"],
            "impact":         c["impact"],
            "confidence":     _explain_confidence(c),
            "source":         "structured + notes",
        })

    # Named flags (copy-forward, medication discontinuation)
    for f in named_flags:
        ft = f.get("flag_type","")
        if ft == "medication_copy_forward":
            issue = {
                "conflict_type":  "medication_copy_forward",
                "severity":       f.get("severity","high"),
                "priority_score": 8,
                "problem":        f"{f.get('medication','medication').title()} listed as active "
                                  f"after patient reported stopping it",
                "evidence": [
                    f"Note {f.get('discontinued_note','')}: patient reported stopping "
                    f"{f.get('medication','')}",
                    f"Note {f.get('reappeared_note','')}: medication listed as active — "
                    f"no re-initiation documented",
                    f"{f.get('note_ids','')}",
                ],
                "action":     f"Verify current {f.get('medication','')} status at this visit. "
                              f"If discontinued, update medication list and document reason. "
                              f"If restarted, document re-initiation date and indication.",
                "impact":     "Inaccurate medication list affects care decisions, medication "
                              "reconciliation, and downstream coding. "
                              "If diabetes medication discontinued, active diabetes management "
                              "coding may not be defensible under MEAT criteria.",
                "confidence": _explain_confidence({"conflict_type": "medication_copy_forward"}),
                "source":     "notes (copy-forward pattern)",
            }
            all_issues.append(issue)
        elif ft == "acute_complication_persistence":
            issue = {
                "conflict_type":  "acute_complication_persistence",
                "severity":       f.get("severity","moderate"),
                "priority_score": 5,
                "problem":        f"{f.get('complication','complication').title()} persisting "
                                  f"{f.get('days_persistent',0)} days — beyond expected window",
                "evidence": [
                    f"First documented: {f.get('note_ids','').split('->')[0].strip()}",
                    f"Still present: {f.get('note_ids','').split('->')[-1].strip()}",
                    f"Expected resolution: ~{f.get('expected_max_days','')} days. "
                    f"No resolution documented.",
                ],
                "action":     f"Verify {f.get('complication','')} is still clinically active. "
                              f"If resolved, document resolution date and remove from active list. "
                              f"If ongoing, document current management.",
                "impact":     "Persistent acute complication documentation without resolution "
                              "suggests copy-forward. May mislead clinical decision support "
                              "and downstream coding systems.",
                "confidence": _explain_confidence({"conflict_type": "acute_complication_persistence"}),
                "source":     "notes (temporal pattern)",
            }
            all_issues.append(issue)

    # Care gaps
    for g in care_gaps:
        gt = g.get("gap_type","")
        if gt == "active_management_gap":
            issue = {
                "conflict_type":  "active_management_gap",
                "severity":       g.get("severity","high"),
                "priority_score": 9,
                "problem":        g["detail"].split(".")[0],
                "evidence": [
                    f"Condition '{g.get('condition','').title()}' documented across multiple notes",
                    f"Medications discontinued: {', '.join(g.get('discontinued_meds',[]))}",
                    "No replacement medication documented in record",
                ],
                "action":     f"Address {g.get('condition','')} management at this visit. "
                              f"Document current treatment, reason for no pharmacotherapy, "
                              f"or initiate appropriate medication. "
                              f"Ensure Treatment element of MEAT criteria is documented.",
                "impact":     "Unmanaged chronic condition creates both care quality risk and "
                              "HCC coding risk. Under CMS-HCC V28, chronic conditions require "
                              "active management documentation. Revenue at risk: "
                              "~$3,000/member/year in RAF leakage if HCC not supported.",
                "confidence": _explain_confidence({"conflict_type": "active_management_gap"}),
                "source":     "notes (management pattern)",
            }
            all_issues.append(issue)
        elif gt == "new_diagnosis_no_followup":
            issue = {
                "conflict_type":  "new_diagnosis_no_followup",
                "severity":       g.get("severity","moderate"),
                "priority_score": 6,
                "problem":        f"New diagnosis of {g.get('diagnosis','').title()} — "
                                  f"no follow-up, no coding, no treatment plan",
                "evidence": [
                    f"Documented in {g.get('note_id','')} ({g.get('date','')})",
                    "No subsequent mention in record",
                    "No ICD code assigned",
                    "No treatment plan documented",
                ],
                "action":     f"Confirm {g.get('diagnosis','')} diagnosis at this visit. "
                              f"If confirmed: add to problem list, assign ICD code, "
                              f"document treatment plan. "
                              f"If resolved or erroneous: document disposition.",
                "impact":     "New diagnosis without follow-up represents either a documentation "
                              "gap or lost-to-follow-up. If chronic condition, "
                              "uncoded diagnoses directly reduce HCC RAF scores.",
                "confidence": _explain_confidence({"conflict_type": "new_diagnosis_no_followup"}),
                "source":     "notes",
            }
            all_issues.append(issue)

    # ICD divergence (depression uncoded)
    for u in icd_gaps.get("free_text_only",[]):
        issue = {
            "conflict_type":  "undocumented_condition",
            "severity":       "moderate",
            "priority_score": 6,
            "problem":        f"{u.get('diagnosis','').title()} in clinical notes — "
                              f"absent from coded record",
            "evidence": [
                f"Appears in clinical note free text: '{u.get('diagnosis','')}'",
                "Not present in structured problem list",
                "No ICD code assigned — invisible to downstream analytics",
            ],
            "action":     f"Add {u.get('diagnosis','')} to structured problem list "
                          f"with appropriate ICD-10 code. "
                          f"Verify active management is documented.",
            "impact":     "Uncoded diagnoses are invisible to ICD-based population health "
                          "queries, care gap detection, and HCC risk adjustment. "
                          "Direct RAF score impact if condition maps to an HCC category.",
            "confidence": _explain_confidence({"conflict_type": "undocumented_condition"}),
            "source":     "notes vs. structured record",
        }
        all_issues.append(issue)

    # Temporal contradictions
    for c in contradictions:
        issue = {
            "conflict_type":  "timeline_inconsistency",
            "severity":       c.get("severity","moderate"),
            "priority_score": 4,
            "problem":        c.get("note",""),
            "evidence":       [c.get("note","")],
            "action":         "Review clinical timeline for this complication. "
                              "Verify procedure documentation is complete and correctly dated.",
            "impact":         "Timeline inconsistencies in surgical documentation may affect "
                              "care coordination and coding accuracy.",
            "confidence":     "Moderate — temporal ordering rule violation detected. "
                              "Clinical review recommended.",
            "source":         "notes (temporal analysis)",
        }
        all_issues.append(issue)

    # Deduplicate by conflict_type — keep highest priority per type
    seen_types = {}
    for issue in sorted(all_issues, key=lambda x: x["priority_score"], reverse=True):
        ct = issue["conflict_type"]
        if ct not in seen_types:
            seen_types[ct] = issue

    ranked = sorted(seen_types.values(), key=lambda x: x["priority_score"], reverse=True)
    return ranked[:3]
