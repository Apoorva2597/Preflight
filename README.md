# EHR Temporal Validator
### Copy-forward detection and longitudinal contradiction flagging in clinical notes

---

## The problem

Chart Awareness and longitudinal summarization tools are meaningful steps forward in clinical AI. But they share a foundational assumption: that the chart is worth trusting.

In practice, it often isn't.

As a Physical Therapist writing ICU and post-operative notes, I noticed that even careful clinicians struggled to reconstruct a patient's longitudinal story from prior documentation. Not because the notes were careless — but because EHR architecture is encounter-first, not patient-first. Each note is written to document *this visit*, not to maintain a coherent timeline.

The result:
- **Copy-forward propagation** — a diagnosis documented in visit 1 silently persists across visits 2 through 12, even after it resolves or changes
- **Stale medications** — active med lists that haven't reflected reality for months
- **Free-text diagnoses** — conditions documented narratively but never coded, invisible to ICD-based longitudinal queries
- **Temporal contradictions** — a complication mentioned without a corresponding procedure, or a treatment that precedes the condition it treats

When a downstream AI system ingests this chart for summarization or coding support, it inherits all of these errors — and presents them with the confidence of a system that has "read the full record."

This project builds the validation layer that should sit beneath longitudinal summarization: **does the chart tell a temporally consistent story?**

---

## What this pipeline does

Given a sequence of clinical notes for a single patient, the pipeline:

1. **Anchors a temporal backbone** — extracts dated clinical events (admissions, procedures, diagnoses, discharge) and orders them into a patient timeline
2. **Extracts clinical entities from free text** — diagnoses, procedures, medications, complications — using rule-based NLP and BioClinicalBERT, including entities that were never structured or coded
3. **Detects copy-forward propagation** — flags entities that appear verbatim or near-verbatim across consecutive notes with no documented clinical basis for persistence
4. **Validates temporal consistency** — checks that complications follow their associated procedures, treatments follow their associated diagnoses, and medication lists reflect documented changes
5. **Compares free-text extraction to ICD codes** — surfaces diagnoses present in narrative text but absent from the coded record, and vice versa
6. **Outputs a trust-annotated timeline** — a per-patient longitudinal view with confidence flags on each entity

---

## Real-world grounding

This pipeline implements an original diagnostic confidence scoring architecture designed by the author — informed by two distinct real-world research contexts:

**Michigan Medicine (2025):** Rule-based NLP pipeline extracting surgical staging from 12,000+ breast reconstruction clinical notes, constructing temporal timelines of surgical events, and extracting complications and treatments relative to surgical stage. That work revealed a consistent pattern: when a diagnosis or complication appeared only in free text and was never coded, it was frequently missed by ICD-based longitudinal queries — yet often clinically significant.

**Production health platform context (2025–2026):** Designed a diagnostic reliability scoring architecture for a consumer health platform aggregating multi-source EHR records — where the core challenge was distinguishing legitimate clinical persistence from erroneous copy-forward propagation across institutional feeds. This informed the weighted composite confidence model implemented here.

The composite scoring formula — `Base Score × Corroboration Multiplier × Temporal Decay Factor × (1 − Contradiction Penalty)` — is an independently derived framework, not a specific system implementation. Decay constants are calibrated from published clinical literature on condition resolution rates (Fortin et al. 2012; van den Bussche et al. 2011; Singh et al. 2013).

**This repository is a public architecture demonstration**, validated on synthetic notes with known ground truth. MIMIC-III integration is supported with credentialed PhysioNet access (see `data/README.md`).

---

## Pipeline architecture

```
clinical_notes (per patient, time-ordered)
        │
        ▼
┌─────────────────────┐
│  1. Temporal        │  Extract dates, events, admissions
│     Anchoring       │  → patient_timeline.json
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Entity          │  Rule-based regex + BioClinicalBERT NER
│     Extraction      │  → diagnoses, procedures, meds, complications
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Composite       │  Base Score × Corroboration Multiplier
│     Confidence      │  × Temporal Decay × (1 − Contradiction Penalty)
│     Scoring         │  Condition-category-specific decay (7 categories)
└────────┬────────────┘  Copy-forward suspicion weighted by condition type
         │               → scored_entities.json (per-entity confidence + audit trail)
         ▼
┌─────────────────────┐
│  4. Temporal        │  Procedure → complication ordering
│     Validation      │  Treatment → diagnosis ordering
└────────┬────────────┘  Resolved diagnosis reappearance detection
         │
         ▼
┌─────────────────────┐
│  5. ICD Divergence  │  Free-text entities vs. coded fields
│     Analysis        │  → coding_gap_report.json
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Trust-Annotated │  Per-patient timeline with confidence scores,
│     Timeline Output │  classifications, and audit trail
└─────────────────────┘  → timeline_output.html / .json
```

### Confidence scoring — key design decisions

**Why multiplicative, not additive?** Additive scoring allows a single strong component (e.g. high base score) to dominate the final result. Multiplicative interaction ensures that a highly reliable source with a strong temporal contradiction still produces a low composite score.

**Why condition-specific decay?** A seroma persisting 6 months post-surgery is suspicious. Hypertension persisting 6 months is expected. A single global decay rate cannot distinguish the two. Seven condition categories with empirically derived decay constants handle this.

**Why is chronic stable hypertension not flagged by copy-forward detection?** The copy-forward suspicion score is weighted by the condition's decay constant (λ). Hypertension (λ=0.04) gets a suspicion weight of ~0.11. Seroma (λ=0.28) gets a suspicion weight of ~0.80. Same text similarity → very different suspicion scores. That's the design.

**What the pipeline cannot determine autonomously:** Whether a chronic condition's persistence is legitimate or erroneous ultimately requires clinical judgment. The pipeline surfaces the cases that warrant review — it does not make the final call. This is intentional.

---

## Quickstart

```bash
git clone https://github.com/apoorvakolhatkar/ehr-temporal-validator
cd ehr-temporal-validator
pip install -r requirements.txt

# Run on synthetic notes (included)
python src/pipeline.py --input data/synthetic/sample_patient_notes.json --output outputs/

# Run on MIMIC-III (requires credentialed access)
python src/pipeline.py --input /path/to/mimic/noteevents.csv --mimic --output outputs/
```

---

## Evaluation design

The pipeline is evaluated on synthetic notes with **known ground truth** — deliberately constructed to represent documented EHR pathologies (copy-forward, stale meds, free-text burial). This is a **controlled methodology demonstration**, not a generalizability claim.

| Metric | What it measures |
|--------|-----------------|
| Entity extraction F1 | Precision/recall vs. annotated ground truth |
| Copy-forward detection rate | % of propagated entities correctly flagged |
| Temporal contradiction rate | % of ordering violations detected |
| ICD divergence recall | % of free-text diagnoses missing from coded record |
| False positive rate | % of legitimate persistence incorrectly flagged |

**Honest limitation:** Synthetic ground truth introduces selection bias — the pipeline catches what we built it to catch. Real-world validation on MIMIC-III or a credentialed clinical dataset is the necessary next step. The methodology section documents what that evaluation would look like.

---

## Limitations and validation roadmap

**Current limitations:**
- Synthetic data only — no claim of generalizability to real EHR populations
- Rule-based temporal anchoring is brittle to non-standard date formats and implicit temporal references ("last week", "at prior visit")
- Copy-forward detection via cosine similarity has high false positive risk for legitimately stable chronic conditions
- BioClinicalBERT NER performance varies significantly by specialty

**Validation roadmap:**
1. MIMIC-III discharge summaries (PhysioNet credentialed access) — diverse patient population, real copy-forward patterns
2. Specialty-stratified evaluation — copy-forward behavior differs significantly between primary care, oncology, and surgical notes
3. Human clinical annotation — ground truth for temporal contradiction detection requires clinician review, not automated labeling
4. Longitudinal stability analysis — distinguishing legitimate persistence (stable chronic condition) from problematic copy-forward (resolved condition persisting)

---

## Project structure

```
ehr-temporal-validator/
├── README.md
├── requirements.txt
├── src/
│   ├── pipeline.py           # Main orchestrator
│   ├── temporal_anchor.py    # Date/event extraction
│   ├── entity_extractor.py   # Rule-based + BioClinicalBERT NER
│   ├── copy_forward.py       # Propagation detection
│   ├── temporal_validator.py # Ordering consistency checks
│   ├── icd_divergence.py     # Free-text vs. coded comparison
│   └── timeline_output.py    # Visualization and reporting
├── data/
│   ├── README.md             # MIMIC-III access instructions
│   └── synthetic/
│       └── sample_patient_notes.json
├── notebooks/
│   └── methodology_walkthrough.ipynb
├── tests/
│   └── test_pipeline.py
└── outputs/
```

---

## Why this matters for ambient AI

The clinical AI stack is moving fast. Ambient scribing is largely solved at the encounter level. Longitudinal summarization (Chart Awareness, Patient Recap, Conditions Advisor) is the current frontier.

The next unsolved problem is **chart trustworthiness** — not summarizing the record, but knowing which parts of it to trust before summarizing.

A Conditions Advisor that surfaces a secondary diagnosis based on a copy-forwarded note from 8 months ago is not improving care. It's systematizing a documentation error at scale.

This pipeline is an early attempt at the validation layer that longitudinal AI systems will eventually need.

---

## Author

**Apoorva Kolhatkar** — Clinical data scientist, Certified Professional Coder, Physical Therapist
MHI Candidate, University of Michigan (May 2026)
[LinkedIn](https://linkedin.com/in/apoorvakolhatkar) · [apokol@umich.edu](mailto:apokol@umich.edu)

*Built as a public methodology demonstration grounded in clinical NLP research at Michigan Medicine.*
