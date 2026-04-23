# Preflight
**EHR Temporal Validator — Pre-Generation Chart Consistency Engine**

---

## Why this exists

I spent years as a Physical Therapist writing clinical notes. A significant portion of that time was not spent on patient care — it was spent reconstructing what had actually happened to a patient by reading through a record that was never designed to tell a coherent story.

EHR architecture is encounter-first. Each note documents *this visit*. The longitudinal picture — what medications are actually active, which diagnoses have resolved, what changed between visit 4 and visit 8 — has to be assembled manually, and it frequently isn't. The result is copy-forward propagation, stale medication lists, and conditions documented only in free text that never make it into the coded record.

This matters now because clinical AI systems are being built on top of that data. Ambient scribing tools, chart summarization systems, and conditions advisors all ingest the EHR and generate output. If the chart is internally inconsistent — if metformin is listed as active three notes after the patient stopped taking it — the AI system inherits that error and presents it with the confidence of a system that has "read the full record."

Preflight is an attempt at the validation layer that sits beneath longitudinal generation: **before you summarize the chart, check whether the chart is worth trusting.**

---

## What the pipeline does

Given a sequence of clinical notes for a single patient, Preflight runs a 10-stage pipeline:

```
Clinical notes (time-ordered)
        │
        ▼
 1. Temporal Anchoring       Extract dates, events, procedures → patient timeline
        │
        ▼
 2. Entity Extraction        GLiNER NER (3-tier graceful degradation)
                             Diagnoses, medications, procedures, complications
        │
        ▼
 2b. LLM Status Resolution   Ollama (llama3.2) resolves longitudinal entity status
                             via RAG over prior notes — is metformin actually active?
        │
        ▼
 3. Confidence Scoring       Composite: Base × Corroboration × Temporal Decay
                             × (1 − Contradiction Penalty)
                             Condition-specific decay constants (7 categories)
        │
        ▼
 4. Temporal Validation      Procedure → complication ordering
                             Treatment → diagnosis ordering
                             Resolved condition reappearance
        │
        ▼
 5. ICD Divergence           Free-text entities vs. coded fields
                             Surfaces diagnoses present in notes, absent from record
        │
        ▼
 6. Freshness Analysis       Per-condition staleness — when was this last
                             clinically re-assessed?
        │
        ▼
 7. Care Gap Detection       Missing expected follow-up given documented conditions
        │
        ▼
 8. Named Flag Detection     Medication conflicts, anticoagulant overlaps,
                             high copy-forward suspicion entities
        │
        ▼
 9. Fusion Analysis          Structured data vs. free-text conflicts
        │
        ▼
10. Top 3 Issues + Trust     Ranked findings with structured signal output
    Score Output             Chart reliability score (0–1)
```

**Output is a structured signal, not a directive.** Each finding includes a recommended signal — `{"action": "suppress", "target": "warfarin", "reason": "transition_documented"}` — for the consuming AI system to act on. Preflight surfaces the risk; the downstream system decides what to do with it.

---

## Key design decisions

**Condition-specific temporal decay.** A seroma persisting 6 months post-surgery is suspicious. Hypertension persisting 6 months is expected. A global decay rate cannot distinguish the two. Seven condition categories with decay constants calibrated from published literature (Fortin et al. 2012; van den Bussche et al. 2011; Singh et al. 2013) handle this. Hypertension (λ=0.04) generates near-zero copy-forward suspicion on stable documentation. Seroma (λ=0.28) flags at high suspicion with the same text similarity.

**Multiplicative confidence, not additive.** Additive scoring lets a strong base score mask a real contradiction. Multiplicative interaction ensures a highly corroborated entity with a documented temporal violation still produces a low composite score.

**LLM layer is swap-able by design.** `OllamaResolver` implements an abstract `LLMResolver` interface. Replacing Ollama with a Claude API call, an Azure OpenAI in-VPC deployment, or a fine-tuned clinical model is a one-class change. No pipeline logic touches the resolver directly.

**Graceful degradation throughout.** If GLiNER is unavailable, the pipeline falls back to BioClinicalBERT, then to rule-based regex. If Ollama is unavailable, GLiNER status labels are used directly. No stage fails silently.

---

## Quickstart

```bash
git clone https://github.com/Apoorva2597/ehr-temporal-validator
cd ehr-temporal-validator
pip install -r requirements.txt

# Ollama (optional — enables LLM status resolution)
# Install from https://ollama.com, then:
ollama pull llama3.2

# Run on included synthetic cohort
python pipeline.py --input data/synthetic/expanded_patient_notes.json

# Run the API
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

---

## API

```http
POST /validate
Content-Type: application/json

{
  "patient_id": "PT_001",
  "notes": [
    { "note_id": "note_1", "date": "2023-03-15", "category": "Progress Note",
      "text": "Patient seen for follow-up. Metformin 500mg daily, continuing." },
    { "note_id": "note_5", "date": "2023-07-22", "category": "Progress Note",
      "text": "Patient reports stopping metformin 3 months ago due to GI side effects." }
  ],
  "icd_codes": ["E11.9", "I10"]
}
```

Returns: chart trust score, top 3 issues, ICD divergence summary, temporal contradictions, named flags, record freshness.

---

## Where Preflight fits in the clinical AI stack

The current state of clinical AI hallucination research is almost entirely focused on **downstream detection** — catching what the model invented after it generated output. VeriFact (NEJM AI, 2025), Abridge's confabulation elimination system, and LLM-as-a-Judge frameworks all operate on generated text, checking claims against source material.

These are important. But they share a blind spot: **they assume the source material is trustworthy.**

Copy-forward propagation means an AI summarization system can generate a perfectly faithful summary of documentation that was clinically wrong for months before the AI touched it. A 2025 npj Digital Medicine study found a 1.47% hallucination rate in LLM clinical note generation — but that metric doesn't capture accurate-but-stale content that was wrong at the source. That failure mode is invisible to downstream detection.

Preflight addresses the upstream problem. It validates chart consistency before generation runs, so that downstream systems are working from the most reliable version of the record available.

The two approaches are complementary, not competing:
- **Upstream (Preflight):** Did the chart tell a consistent story before we summarized it?
- **Downstream (VeriFact, Abridge):** Did the model accurately represent what the chart said?

Both are necessary. Neither is sufficient alone.

---

## Honest evaluation framing

This pipeline is demonstrated on a synthetic cohort of 4 patients, deliberately constructed to exhibit the full range of failure modes the system detects — copy-forward, stale medications, temporal ordering violations, uncoded diagnoses.

**What this validates:** The architecture correctly identifies planted errors in a controlled setting. The pipeline components — entity extraction, temporal reasoning, ICD divergence — behave as designed.

**What this does not validate:** False positive rates on clean records with legitimate clinical stability. Recall on real clinical language, which is highly variable by specialty and institution. Generalizability to real EHR populations.

**What honest real-world validation would look like:** MIMIC-III discharge summaries (40,000+ ICU records, PhysioNet credentialed access) with specialty-stratified evaluation and clinician annotation for ground truth on temporal contradiction detection. That is the defined next step — the architecture is designed with that evaluation in mind.

The trust score is a relative risk signal within a cohort, not a validated clinical accuracy metric. Penalty weights are calibrated to synthetic ground truth and would require recalibration against annotated real data.

---

## Limitations

- **Synthetic data only** — no generalizability claim to real EHR populations
- **Copy-forward detection** has elevated false positive risk for legitimately stable chronic conditions (mitigated by condition-specific decay constants, not eliminated)
- **ICD divergence recall** on real clinical language is unknown — free-text to code mapping using GLiNER works on canonical terms; clinical language variation by specialty and provider is not characterized
- **Temporal anchoring** is brittle to non-standard date formats and implicit references ("at prior visit", "last month")
- **Trust score weights** are heuristic, not derived from clinical outcomes data

---

## Project structure

```
preflight/
├── pipeline.py                    # CLI orchestrator
├── entity_extractor.py            # GLiNER / BioClinicalBERT / regex (3-tier)
├── ollama_resolver.py             # LLM status resolution (implements LLMResolver)
├── llm_resolver.py                # Abstract base class — swap in any LLM backend
├── confidence_scorer.py           # Composite scoring with temporal decay
├── temporal_validator.py          # Ordering consistency checks
├── icd_divergence.py              # Free-text vs. coded comparison
├── longitudinal_state_builder.py  # Pre-filters entities for LLM resolution
├── freshness.py                   # Per-condition staleness analysis
├── care_gaps.py                   # Missing follow-up detection
├── named_flags.py                 # Medication conflicts, copy-forward flags
├── fusion.py                      # Structured vs. free-text conflicts
├── top3_engine.py                 # Ranked issue selection
├── api/
│   ├── main.py                    # FastAPI app with lifespan startup
│   ├── routes.py                  # /validate, /health, /schema
│   ├── services.py                # Pipeline adapter (shared model instances)
│   └── schemas.py                 # Pydantic I/O models
├── data/
│   └── synthetic/
│       └── expanded_patient_notes.json   # 4-patient synthetic cohort
├── condition_taxonomy.yaml        # Decay constants and condition categories
├── preflight_demo.html                  # Interactive demo — open in browser
└── requirements.txt
```

---

## Background

**Apoorva Kolhatkar** — Physical Therapist · Certified Professional Coder · MHI Candidate, University of Michigan (May 2026)

Clinical NLP pipeline work at Michigan Medicine (2025): rule-based extraction of surgical staging from 12,000+ breast reconstruction notes, longitudinal complication and treatment timelines relative to surgical stage.

This project is a public architecture demonstration. It is not a clinical product and has not been validated for clinical use.

[LinkedIn](https://linkedin.com/in/apoorvakolhatkar) · [apokol@umich.edu](mailto:apokol@umich.edu)
