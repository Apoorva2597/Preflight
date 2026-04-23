"""
freshness.py
Computes documentation freshness per note and across the record.

Core insight: EHR copy-forward means notes that look comprehensive
may contain almost no new clinical information. This module quantifies
how much genuine new content each note contributes vs. what was
copied from the prior note.

Freshness score per note = 1 - similarity_to_prior_note
Record freshness = weighted average, weighted by note position
  (later notes carry more weight — deteriorating freshness over time
   is more clinically concerning than early copy-forward)

Output per note:
  - freshness_score: 0-1 (1 = entirely new content, 0 = pure copy)
  - new_sentences: list of sentences unique to this note
  - copied_sentences: list of sentences copied from prior note
  - classification: Fresh / Updating / Stagnant / Copy
"""

import re
from difflib import SequenceMatcher


CLINICAL_SECTION_PATTERNS = [
    r"(?:assessment|plan|impression|diagnosis|medications?|problem\s+list|"
    r"history\s+of\s+present\s+illness|hpi|review\s+of\s+systems|ros|"
    r"physical\s+exam|pe|vitals?|labs?|imaging|procedures?)",
]

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.lower())
    return [s.strip() for s in sentences if len(s.strip()) > 25]

def sentence_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_copied(sentence, prior_sentences, threshold=0.82):
    return any(sentence_sim(sentence, ps) >= threshold for ps in prior_sentences)

def classify_freshness(score):
    if score >= 0.70: return "Fresh",     "#1a7f5a"
    if score >= 0.45: return "Updating",  "#2563a8"
    if score >= 0.20: return "Stagnant",  "#b06000"
    return               "Copy",      "#c0392b"


class FreshnessAnalyzer:

    def analyze(self, notes):
        """
        Compute per-note freshness and record-level freshness summary.
        """
        note_results = []
        prior_sentences = []

        for i, note in enumerate(notes):
            text = note.get("text", "")
            curr_sentences = split_sentences(text)

            if not prior_sentences or not curr_sentences:
                score = 1.0
                new_sents = curr_sentences
                copied_sents = []
            else:
                new_sents    = [s for s in curr_sentences if not is_copied(s, prior_sentences)]
                copied_sents = [s for s in curr_sentences if is_copied(s, prior_sentences)]
                total = len(curr_sentences)
                score = round(len(new_sents) / total, 3) if total > 0 else 1.0

            label, color = classify_freshness(score)

            note_results.append({
                "note_id":          note["note_id"],
                "date":             note.get("date", ""),
                "category":         note.get("category", ""),
                "freshness_score":  score,
                "freshness_label":  label,
                "freshness_color":  color,
                "new_sentence_count":    len(new_sents),
                "copied_sentence_count": len(copied_sents),
                "total_sentences":       len(curr_sentences),
                "new_sentences":    new_sents[:3],    # sample only
                "copied_sentences": copied_sents[:3], # sample only
            })

            prior_sentences = curr_sentences  # roll forward

        # Record-level summary
        if note_results:
            # Weight later notes more heavily
            weights = [i + 1 for i in range(len(note_results))]
            total_w = sum(weights)
            weighted_freshness = sum(
                r["freshness_score"] * w
                for r, w in zip(note_results, weights)
            ) / total_w

            stagnant = [r for r in note_results if r["freshness_label"] in ("Stagnant","Copy")]
            fresh    = [r for r in note_results if r["freshness_label"] in ("Fresh","Updating")]
        else:
            weighted_freshness = 1.0
            stagnant = []
            fresh = []

        return {
            "notes":                  note_results,
            "record_freshness_score": round(weighted_freshness, 3),
            "stagnant_note_count":    len(stagnant),
            "fresh_note_count":       len(fresh),
            "total_notes":            len(note_results),
            "summary": (
                f"{len(fresh)} of {len(note_results)} notes contain substantial new content. "
                f"{len(stagnant)} notes appear predominantly copy-forwarded."
            )
        }
