"""
temporal_anchor.py
Extracts dated clinical events from notes and builds a per-patient timeline.
"""

import re
from datetime import datetime
from typing import Optional


# Patterns for clinical event markers
PROCEDURE_PATTERNS = [
    r"\b(underwent|performed|completed|status post|s/p)\s+([a-z\s\-]+(?:surgery|repair|reconstruction|excision|biopsy|procedure|operation|placement|removal|resection|graft|flap|mastectomy|lumpectomy|debridement))",
    r"\b(surgery|procedure|operation)\s+(?:was\s+)?(?:performed|completed|done)\b",
]

ADMISSION_PATTERNS = [
    r"\b(admitted|admission|presented to|transferred from|transferred to)\b",
    r"\b(discharge[d]?|discharged home|discharged to)\b",
]

DIAGNOSIS_EVENT_PATTERNS = [
    r"\b(diagnosed with|diagnosis of|new diagnosis|confirmed diagnosis)\s+([a-z\s\-,]+)",
    r"\b(history of|h/o|hx of)\s+([a-z\s\-,]+)",
]

COMPLICATION_PATTERNS = [
    r"\b(complicated by|complication[s]?|developed|presented with|noted to have)\s+([a-z\s\-,]+(?:infection|wound|seroma|hematoma|necrosis|dehiscence|failure|leak|stenosis|thrombosis|embolism|sepsis|abscess))",
]

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
    r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
]


def parse_date(date_str: str) -> Optional[datetime]:
    formats = [
        "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%m/%d/%y", "%B %d, %Y", "%B %d %Y", "%d %B %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


class TemporalAnchor:

    def build_timeline(self, notes: list[dict]) -> list[dict]:
        """
        Build a chronologically ordered timeline of clinical events
        across all notes for a single patient.
        """
        events = []
        for note in notes:
            note_date = parse_date(note.get("date", ""))
            text = note.get("text", "").lower()
            note_events = self._extract_events(text, note_date, note["note_id"])
            events.extend(note_events)

        # Sort by date, placing undated events at end
        events.sort(key=lambda e: (e["date"] is None, e["date"] or ""))
        # Remove generic/uninformative events that duplicate specific ones
        GENERIC_PATTERNS = [
            "procedure was completed", "procedure completed",
            "procedure was performed", "surgery was performed",
        ]
        events = [
            e for e in events
            if not any(g in e.get("text","").lower() for g in GENERIC_PATTERNS)
        ]
        return events

    def _extract_events(self, text: str, note_date: Optional[datetime], note_id: str) -> list[dict]:
        events = []

        for pattern in PROCEDURE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                events.append({
                    "event_type": "procedure",
                    "text": match.group(0).strip(),
                    "note_id": note_id,
                    "date": note_date.strftime("%Y-%m-%d") if note_date else None,
                    "inferred_date": note_date is None
                })

        for pattern in ADMISSION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                events.append({
                    "event_type": "admission_discharge",
                    "text": match.group(0).strip(),
                    "note_id": note_id,
                    "date": note_date.strftime("%Y-%m-%d") if note_date else None,
                    "inferred_date": note_date is None
                })

        for pattern in COMPLICATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                events.append({
                    "event_type": "complication",
                    "text": match.group(0).strip(),
                    "note_id": note_id,
                    "date": note_date.strftime("%Y-%m-%d") if note_date else None,
                    "inferred_date": note_date is None
                })

        for pattern in DIAGNOSIS_EVENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                events.append({
                    "event_type": "diagnosis",
                    "text": match.group(0).strip(),
                    "note_id": note_id,
                    "date": note_date.strftime("%Y-%m-%d") if note_date else None,
                    "inferred_date": note_date is None
                })

        return events
