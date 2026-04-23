# Preflight — Synthetic Evaluation Dataset

**File:** `data/synthetic/expanded_patient_notes.json`
**Patients:** 4 (SYNTH_001 – SYNTH_004)
**Notes:** 26 total
**Date range:** 2022–2023 (synthetic)

---

## Purpose

This dataset evaluates Preflight's ability to detect four EHR documentation
failure modes. Every planted error replicates a specific pattern documented
in peer-reviewed literature. This is not a generalizability claim — it is
a controlled methodology demonstration showing the pipeline correctly
identifies known error classes.

**All patient names, dates, and clinical details are synthetic.
No real patient data is used.**

---

## Why synthetic replication of documented errors

The choice to replicate published error patterns rather than invent novel
ones is deliberate. It means:

1. The errors are clinically plausible — they reflect how documentation
   failures actually manifest in real EHR systems
2. Detection performance can be interpreted against a known baseline
3. The pipeline is tested against error classes that continue to appear
   on ECRI's annual Top Patient Safety Concerns lists — meaning existing
   EHR safeguards have not solved them

The errors in this dataset are not hypothetical. They are documented,
quantified, and still occurring.

---

## The four failure modes — citations most recent first

---

### Error Type 1 — Commission error: discontinued medication persisting on active list

**Planted in:** SYNTH_001 (metformin to sitagliptin transition),
SYNTH_004 (metformin held per nephrology)

**Classification:** Error of commission — present in the record,
not taken by the patient.

**Clinical pattern:** Patient discontinues or is taken off a medication.
The medication persists in the active medication list and in background
documentation across subsequent notes. The new agent or held status is
correctly documented in clinical assessments but the structured list is
never updated. A downstream AI system reading the medication list sees
the discontinued drug as active.

**Evidence:**

ECRI / ISMP, April 2025 — Top Patient Safety Concerns:
Failure to document medication holds and discontinuations during care
transitions named a persistent preventable hazard. ECRI specifically
identified failure to remove discontinued medications from active lists
and over-reliance on medication summaries without active reconciliation.

Francis et al., Health Information Management, 2025:
In a system with a well-established EHR linked to pharmacy dispensing,
medication discrepancies occurred in 60% of ambulatory patients.
Errors of commission — medication present in record, not taken by
patient — were most common in patients on multiple medications.

AMA STEPS Forward analysis, 2022 (100 million+ EHR notes):
50.1% of all note text was duplicated from prior documentation.
By 2020, copied text had grown to 54.2% of note content — more than
half of what clinicians read in a chart may reflect older documentation
rather than current clinical status.

PMC4476907 (foundational, University of Michigan):
More than 40% of medication errors traceable to inadequate reconciliation.
Up to 60% of admitted patients have at least one reconciliation error
despite mandatory reconciliation processes. Electronic tools often lack
functionality to accurately reconcile medications.

**SYNTH_001 design:** Patient self-discontinues metformin (note_5,
May 2023), switched to sitagliptin. Clinical assessments in notes 6,
7, 8 correctly reference sitagliptin. Metformin 1000mg BID remains on
the active medication list — never reconciled. The error is subtle:
a provider reading only the medication list sees metformin as active.
A provider reading the SOAP assessment sees sitagliptin. Preflight
detects the inconsistency between these two information streams.

**SYNTH_004 design:** Metformin held at discharge per nephrology
(note_2) due to elevated creatinine. Reappears as active on structured
medication list from note_4 onward. Patient tells cardiologist in note_6
that nephrology instructed discontinuation — the cardiologist's note
then references metformin as active per the medication list.
Structured record and narrative are contradictory within the same note.

---

### Error Type 2 — Anticoagulant transition copy-forward

**Planted in:** SYNTH_003 (warfarin to apixaban)

**Clinical pattern:** Documented warfarin-to-apixaban transition
(note_4) not reflected in subsequent notes. Warfarin continues as
the stated anticoagulant in notes 5 and 6. No INR monitoring referenced
in the copy-forwarded notes — consistent with a provider who has
transitioned to a DOAC (no INR required) but whose notes carry forward
prior warfarin documentation unchanged.

**Evidence:**

ECRI / ISMP, 2025:
Anticoagulant transitions and medication list accuracy at care
transitions named among top persistent medication safety hazards.

Pennsylvania Patient Safety Reporting System (ISMP), 2015:
Analysis of 831 oral anticoagulant errors — incomplete medication
lists were the second most common error source, concentrated at
transitions of care. High-alert medications including anticoagulants
carry heightened risk when lists are inaccurate.
This foundational report continues to be cited in 2025 ISMP guidance
because the underlying problem has not been resolved.

**Design note — no INR values in copy-forwarded notes:**
INR monitoring is warfarin-specific. Including fabricated INR values
after a documented DOAC transition would represent a fabricated lab
value — a different and worse error class. The planted error is the
medication copy-forward only. Absence of INR in notes 5 and 6 is
itself a signal: a provider managing active warfarin therapy would
document INR. A provider who copy-forwarded warfarin without actually
managing it would not.

---

### Error Type 3 — Diagnosis in free text, absent from coded record

**Planted in:** SYNTH_001 (depression), SYNTH_004 (diabetic nephropathy)

**Clinical pattern:** Clinically significant diagnosis documented in
free-text notes, never added to the structured problem list or assigned
an ICD code. AI systems using structured data as primary source will
not surface this diagnosis in generated summaries, care gap analysis,
or quality measures.

**Evidence:**

Swiss Medical Weekly, February 2025:
Cross-sectional study of 6,000 patients across 10 general practices
found diagnoses frequently documented as unstructured free-text entries.
Prevalence estimates for common conditions varied significantly by
whether free-text diagnoses were included alongside structured data.

AHRQ — Documenting Diagnosis, 2024 (reviewed):
Approximately 80% of EHR data is unstructured text. Essential clinical
details are frequently hidden in notes rather than captured in
structured fields.

PMC9759969 — EHR Problem List Audit, 2021:
One year after Epic implementation at a major teaching hospital, many
secondary data returns still relied on ICD-10 codes entered
retrospectively by coding staff rather than by clinicians at point
of care. Structured problem list use was not consistent.

**SYNTH_001 design:** Depression documented in note_6 with PHQ-9
score 12, psychiatry referral, sertraline initiated. Never added to
ICD codes (C50.912, Z90.11, E11.9, I10). Any AI system using the
structured record as primary source omits depression entirely.

**SYNTH_004 design:** Diabetic nephropathy identified by nephrology
(note_2) and confirmed in follow-up. Never appears in primary care ICD
codes. CKD stage 3 is coded (N18.3) but the etiology — diabetic
nephropathy — lives only in specialist narrative notes.

---

### Error Type 4 — Control case

**Patient:** SYNTH_002

**Purpose:** False positive baseline. Stable chronic disease management
(diabetes, hypertension, hyperlipidemia) across 5 quarterly visits.
Consistent medications. Logical lab trends. No planted errors.

**Expected pipeline behavior:** Zero named flags. Low copy-forward
suspicion (condition-specific decay λ=0.04 for chronic stable conditions
suppresses suspicion on legitimately stable documentation). High
reliability signal. This tests the pipeline's ability to distinguish
legitimate clinical stability from copy-forward propagation — the
central false positive risk.

---

## Dataset limitations

- Error types and severity were chosen by the pipeline designer
- Synthetic language may not capture real provider documentation variability
- No adversarial test set designed to confuse the pipeline
- No held-out evaluation on notes the designer did not author
- Real-world validation on MIMIC-III against clinician-annotated
  ground truth is the defined next step

---

## Citation index

| Source | Year | Maps to |
|--------|------|---------|
| ECRI / ISMP Top Patient Safety Concerns | 2025 | Error Types 1, 2 |
| Swiss Medical Weekly, free-text diagnosis study | 2025 | Error Type 3 |
| Francis et al., Health Information Management | 2025 | Error Type 1 |
| AMA STEPS Forward, 100M note analysis | 2022 | Error Types 1, 2 |
| AHRQ, Documenting Diagnosis | 2024 | Error Type 3 |
| PMC9759969, EHR problem list audit | 2021 | Error Type 3 |
| PMC4476907, EMR Medication Reconciliation | 2020 | Error Type 1 |
| PA-PSRS anticoagulant error analysis (ISMP) | 2015 | Error Type 2 |

---

*Preflight v0.1.0 — Synthetic cohort — Not validated for clinical use*
*Apoorva Kolhatkar, MHI, University of Michigan*
