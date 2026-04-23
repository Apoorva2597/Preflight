"""
entity_extractor.py — EHR Temporal Validator
Apoorva Kolhatkar | Michigan Medicine NLP Research

Clinical NER pipeline with three-tier graceful degradation:

  Tier 1 — GLiNER (preferred)
    Zero-shot NER using urchade/gliner_mediumv2.1
    Extracts any entity type defined at inference time — no fine-tuning needed.
    Published: Zaratiana et al., NAACL 2024.
    Entities: medication, medication_status, diagnosis, diagnosis_status,
              procedure, complication, temporal_expression, dosage, lab_value

  Tier 2 — BioClinicalBERT NER (fallback)
    samrawal/bert-base-uncased_clinical-ner fine-tuned on i2b2/n2c2 clinical notes.
    Fixed label set. Used when GLiNER is not installed.

  Tier 3 — Rule-based regex (final fallback)
    Deterministic, no dependencies. Covers ~200 clinical terms.
    Suitable for development. Not production-grade.

Architecture note:
    Entity STATUS (active, discontinued, historical) requires contextual reasoning
    beyond NER alone. GLiNER's medication_status and diagnosis_status labels capture
    surface-level cues. Full status resolution is handled downstream by the Ollama
    reasoning layer (ollama_resolver.py) using RAG over prior notes.

Install:
    Tier 1: pip install gliner
            Model: urchade/gliner_mediumv2.1 (downloaded on first use)
    Tier 2: pip install transformers torch
    Tier 3: No additional dependencies
"""

import re
from typing import Optional

# ── Tier 1: GLiNER ───────────────────────────────────────────────────────────
try:
    from gliner import GLiNER as _GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

# ── Tier 2: BioClinicalBERT ──────────────────────────────────────────────────
try:
    from transformers import pipeline as _hf_pipeline, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ── GLiNER entity labels ──────────────────────────────────────────────────────
GLINER_LABELS = [
    "medication",
    "medication_status",
    "diagnosis",
    "diagnosis_status",
    "procedure",
    "complication",
    "temporal_expression",
    "dosage",
    "lab_value",
]

GLINER_THRESHOLD = 0.40

GLINER_TYPE_MAP = {
    "medication":          "medications",
    "medication_status":   "medication_statuses",
    "diagnosis":           "diagnoses",
    "diagnosis_status":    "diagnosis_statuses",
    "procedure":           "procedures",
    "complication":        "complications",
    "temporal_expression": "temporal_expressions",
    "dosage":              "dosages",
    "lab_value":           "lab_values",
}

# ── Rule-based patterns (Tier 3) ─────────────────────────────────────────────

MEDICATION_PATTERNS = [
    r"\b([A-Z][a-z]+(?:in|ol|ate|ide|ine|one|mab|nib|zole|pril|artan|cillin|mycin|cycline))\s+\d+\s*(?:mg|mcg|g|units?|IU)\b",
    r"\b(aspirin|warfarin|heparin|apixaban|rivaroxaban|dabigatran|edoxaban|clopidogrel|ticagrelor|prasugrel)\b",
    r"\b(metoprolol|carvedilol|bisoprolol|atenolol|propranolol|labetalol)\b",
    r"\b(lisinopril|enalapril|ramipril|benazepril|captopril|losartan|valsartan|olmesartan|irbesartan|candesartan)\b",
    r"\b(amlodipine|nifedipine|diltiazem|verapamil|felodipine)\b",
    r"\b(furosemide|torsemide|bumetanide|hydrochlorothiazide|chlorthalidone|spironolactone|eplerenone)\b",
    r"\b(atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|pitavastatin|fluvastatin)\b",
    r"\b(digoxin|amiodarone|dronedarone|flecainide|sotalol)\b",
    r"\b(metformin|insulin|glargine|detemir|degludec|lispro|aspart|glulisine)\b",
    r"\b(sitagliptin|saxagliptin|linagliptin|alogliptin|empagliflozin|dapagliflozin|canagliflozin)\b",
    r"\b(liraglutide|semaglutide|dulaglutide|exenatide|tirzepatide)\b",
    r"\b(glipizide|glyburide|glimepiride|pioglitazone|rosiglitazone)\b",
    r"\b(tiotropium|umeclidinium|aclidinium|glycopyrrolate|ipratropium)\b",
    r"\b(albuterol|levalbuterol|salmeterol|formoterol|indacaterol|vilanterol)\b",
    r"\b(fluticasone|budesonide|beclomethasone|mometasone|ciclesonide)\b",
    r"\b(montelukast|zafirlukast|roflumilast)\b",
    r"\b(sertraline|fluoxetine|paroxetine|escitalopram|citalopram|venlafaxine|duloxetine|bupropion|mirtazapine)\b",
    r"\b(alprazolam|lorazepam|diazepam|clonazepam|buspirone|hydroxyzine)\b",
    r"\b(quetiapine|olanzapine|risperidone|aripiprazole|ziprasidone|haloperidol)\b",
    r"\b(gabapentin|pregabalin|topiramate|levetiracetam|lamotrigine|valproate|carbamazepine)\b",
    r"\b(donepezil|memantine|rivastigmine|galantamine)\b",
    r"\b(oxycodone|hydrocodone|morphine|hydromorphone|tramadol|codeine|fentanyl|buprenorphine)\b",
    r"\b(ibuprofen|naproxen|celecoxib|meloxicam|indomethacin|ketorolac|diclofenac)\b",
    r"\b(acetaminophen|prednisone|methylprednisolone|dexamethasone|triamcinolone)\b",
    r"\b(colchicine|allopurinol|febuxostat|probenecid)\b",
    r"\b(amoxicillin|augmentin|ampicillin|cephalexin|cefazolin|ceftriaxone|cefepime|meropenem|ertapenem)\b",
    r"\b(azithromycin|clarithromycin|erythromycin|doxycycline|minocycline|tetracycline)\b",
    r"\b(ciprofloxacin|levofloxacin|moxifloxacin|trimethoprim|sulfamethoxazole)\b",
    r"\b(vancomycin|linezolid|daptomycin|piperacillin|tazobactam|clindamycin|metronidazole)\b",
    r"\b(omeprazole|pantoprazole|lansoprazole|esomeprazole|rabeprazole|famotidine|ranitidine)\b",
    r"\b(ondansetron|metoclopramide|prochlorperazine|promethazine)\b",
    r"\b(levothyroxine|methimazole|propylthiouracil)\b",
    r"\b(alendronate|risedronate|denosumab|teriparatide)\b",
    r"\b(ferrous\s+sulfate|folic\s+acid|cyanocobalamin|cholecalciferol|ergocalciferol)\b",
]

DIAGNOSIS_PATTERNS = [
    r"\b(diabetes(?:\s+mellitus)?(?:\s+type\s+[12])?|diabetic\s+nephropathy|diabetic\s+neuropathy|diabetic\s+retinopathy|hyperlipidemia|mixed\s+hyperlipidemia|hypothyroidism|hyperthyroidism|obesity|metabolic\s+syndrome)\b",
    r"\b(hypertension|heart\s+failure|congestive\s+heart\s+failure|atrial\s+fibrillation|atrial\s+flutter|coronary\s+artery\s+disease|myocardial\s+infarction|angina|cardiomyopathy)\b",
    r"\b(chronic\s+kidney\s+disease|acute\s+kidney\s+injury|end[\s-]stage\s+renal\s+disease|nephrotic\s+syndrome|glomerulonephritis|cardiorenal\s+syndrome)\b",
    r"\b(COPD|chronic\s+obstructive\s+pulmonary\s+disease|asthma|pulmonary\s+hypertension|pulmonary\s+embolism|pleural\s+effusion|pneumonia|bronchitis|emphysema|sleep\s+apnea)\b",
    r"\b(depression|anxiety|bipolar\s+disorder|schizophrenia|PTSD|dementia|Alzheimer|Parkinson|epilepsy|stroke|TIA|peripheral\s+neuropathy)\b",
    r"\b(breast\s+cancer|invasive\s+ductal\s+carcinoma|invasive\s+lobular\s+carcinoma|ductal\s+carcinoma|lung\s+cancer|colon\s+cancer|prostate\s+cancer|lymphoma|leukemia|melanoma)\b",
    r"\b(GERD|gastroesophageal\s+reflux|peptic\s+ulcer|Crohn|ulcerative\s+colitis|cirrhosis|hepatitis|pancreatitis|diverticulitis|anemia|gout|osteoporosis|rheumatoid\s+arthritis|osteoarthritis)\b",
    r"\b(sepsis|bacteremia|cellulitis|wound\s+infection|UTI|urinary\s+tract\s+infection|Clostridium\s+difficile|C\.\s+diff|osteomyelitis|endocarditis)\b",
    r"\b(deep\s+vein\s+thrombosis|DVT|peripheral\s+vascular\s+disease|aortic\s+aneurysm|carotid\s+stenosis)\b",
]

PROCEDURE_PATTERNS = [
    r"\b(mastectomy|lumpectomy|reconstruction|TRAM\s+flap|DIEP\s+flap|tissue\s+expander|implant\s+placement|sentinel\s+node\s+biopsy|axillary\s+dissection|cholecystectomy|appendectomy|colectomy|hernia\s+repair|hip\s+replacement|knee\s+replacement|spinal\s+fusion|debridement|skin\s+graft)\b",
]

COMPLICATION_PATTERNS = [
    r"\b(wound\s+infection|seroma|hematoma|skin\s+necrosis|wound\s+dehiscence|implant\s+failure|capsular\s+contracture|lymphedema|neuropathy|deep\s+vein\s+thrombosis|pulmonary\s+embolism|urinary\s+tract\s+infection|pneumonia|sepsis|anastomotic\s+leak|ileus|readmission)\b",
]

NEGATION_PATTERNS = [
    r"\b(no|not|without|denies|denied|negative\s+for|rules?\s+out|ruled\s+out|absent|never|no\s+evidence\s+of)\b"
]


def is_negated(text: str, match_start: int, window: int = 6) -> bool:
    preceding = text[:match_start].split()[-window:]
    preceding_text = " ".join(preceding).lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, preceding_text, re.IGNORECASE):
            return True
    return False


def _deduplicate(entities: list) -> list:
    seen = set()
    deduped = []
    for e in entities:
        key = (e["text"].lower().strip(), e.get("negated", False))
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


class EntityExtractor:
    """
    Clinical entity extractor with three-tier NER pipeline.
    Tier selection is automatic based on installed packages.
    """

    def __init__(self, gliner_model: str = "urchade/gliner_mediumv2.1"):
        self.tier = None
        self.gliner = None
        self.ner_pipeline = None

        # Tier 1: GLiNER
        if GLINER_AVAILABLE:
            try:
                self.gliner = _GLiNER.from_pretrained(gliner_model)
                self.tier = 1
                print(f"[EntityExtractor] Tier 1 — GLiNER: {gliner_model}")
            except Exception as e:
                print(f"[EntityExtractor] GLiNER unavailable: {e}")

        # Tier 2: BioClinicalBERT
        if self.tier is None and TRANSFORMERS_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.ner_pipeline = _hf_pipeline(
                    "ner",
                    model="samrawal/bert-base-uncased_clinical-ner",
                    tokenizer=tokenizer,
                    aggregation_strategy="simple"
                )
                self.tier = 2
                print("[EntityExtractor] Tier 2 — BioClinicalBERT NER")
            except Exception as e:
                print(f"[EntityExtractor] BioClinicalBERT unavailable: {e}")

        # Tier 3: Rule-based
        if self.tier is None:
            self.tier = 3
            print("[EntityExtractor] Tier 3 — rule-based. pip install gliner for Tier 1.")

    def extract_all(self, notes: list) -> list:
        return [
            {
                "note_id": note["note_id"],
                "date": note.get("date"),
                "extraction_tier": self.tier,
                "entities": self._extract_from_note(note)
            }
            for note in notes
        ]

    def _extract_from_note(self, note: dict) -> dict:
        text = note.get("text", "")
        if self.tier == 1:
            return self._gliner_extract(text)
        elif self.tier == 2:
            return self._bioclinicalbert_extract(text)
        else:
            return self._rule_extract_all(text)

    def _gliner_extract(self, text: str) -> dict:
        result = {bucket: [] for bucket in GLINER_TYPE_MAP.values()}
        try:
            entities = self.gliner.predict_entities(
                text, GLINER_LABELS, threshold=GLINER_THRESHOLD
            )
            for ent in entities:
                label = ent.get("label", "")
                bucket = GLINER_TYPE_MAP.get(label)
                if not bucket:
                    continue
                span_text = ent.get("text", "").strip()
                score = round(ent.get("score", 0.0), 3)
                start = ent.get("start", 0)
                negated = False
                if label in ("medication", "diagnosis"):
                    negated = is_negated(text, start)
                result[bucket].append({
                    "text": span_text,
                    "type": label,
                    "score": score,
                    "start": start,
                    "negated": negated,
                    "source": "gliner",
                })
            for bucket in result:
                result[bucket] = _deduplicate(result[bucket])
            # Rule-based augment for procedures/complications
            rule_proc = self._rule_extract(text, PROCEDURE_PATTERNS, "procedure")
            rule_comp = self._rule_extract(text, COMPLICATION_PATTERNS, "complication")
            result["procedures"] = _deduplicate(result["procedures"] + rule_proc)
            result["complications"] = _deduplicate(result["complications"] + rule_comp)
        except Exception as e:
            print(f"[EntityExtractor] GLiNER error: {e}. Falling back to rule-based.")
            return self._rule_extract_all(text)
        return result

    def _bioclinicalbert_extract(self, text: str) -> dict:
        result = self._rule_extract_all(text)
        try:
            words = text.split()
            chunks = [
                " ".join(words[i:i+400])
                for i in range(0, len(words), 400)
            ]
            for chunk in chunks:
                for r in self.ner_pipeline(chunk):
                    if r["score"] < 0.75:
                        continue
                    group = r.get("entity_group", "").lower()
                    span = r.get("word", "").strip()
                    if "problem" in group:
                        result["diagnoses"].append({
                            "text": span, "type": "diagnosis",
                            "score": round(r["score"], 3),
                            "negated": False, "source": "bioclinicalbert"
                        })
                    elif "treatment" in group:
                        result["medications"].append({
                            "text": span, "type": "medication",
                            "score": round(r["score"], 3),
                            "negated": False, "source": "bioclinicalbert"
                        })
            for bucket in ("medications", "diagnoses"):
                result[bucket] = _deduplicate(result[bucket])
        except Exception as e:
            print(f"[EntityExtractor] BioClinicalBERT error: {e}")
        return result

    def _rule_extract_all(self, text: str) -> dict:
        return {
            "medications":          self._rule_extract(text, MEDICATION_PATTERNS, "medication"),
            "diagnoses":            self._rule_extract(text, DIAGNOSIS_PATTERNS, "diagnosis"),
            "procedures":           self._rule_extract(text, PROCEDURE_PATTERNS, "procedure"),
            "complications":        self._rule_extract(text, COMPLICATION_PATTERNS, "complication"),
            "medication_statuses":  [],
            "diagnosis_statuses":   [],
            "temporal_expressions": [],
            "dosages":              [],
            "lab_values":           [],
        }

    def _rule_extract(self, text: str, patterns: list, entity_type: str) -> list:
        entities = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                negated = is_negated(text, match.start())
                entities.append({
                    "text": match.group(0).strip(),
                    "type": entity_type,
                    "start": match.start(),
                    "negated": negated,
                    "source": "rule_based",
                })
        return _deduplicate(entities)
