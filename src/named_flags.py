"""
named_flags.py
Detects specific, clinically meaningful named flags.

Quick-win flags:
  1. Medication listed as active after patient reported stopping it
  2. Acute complication persisting beyond expected resolution window
"""

import re
from datetime import datetime

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

RESOLUTION_PHRASES = [
    r"\b(resolved|resolving|healed|cleared|improved|no\s+longer\s+present|no\s+evidence\s+of)\b",
]

ACUTE_COMPLICATIONS = {
    "seroma": 42,
    "hematoma": 28,
    "wound infection": 21,
    "wound dehiscence": 56,
    "ileus": 14,
}


def parse_date(s):
    if not s: return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
        try: return datetime.strptime(s, fmt)
        except: continue
    return None


def months_between(d1, d2):
    if not d1 or not d2: return 0.0
    return abs((d2 - d1).days) / 30.44


class NamedFlagDetector:

    def detect(self, notes, extracted_entities):
        flags = []
        flags.extend(self._check_medication_reappearance(notes, extracted_entities))
        flags.extend(self._check_acute_persistence(notes))
        return flags

    def _check_medication_reappearance(self, notes, extracted_entities):
        """
        Medication reported as stopped in note N, then listed as active
        in a later note without re-initiation. Classic copy-forward error.
        Demonstrated case: metformin discontinued in SYNTH_001 note_5,
        copy-forwarded as active in note_6.
        """
        flags = []
        discontinued = {}  # med_name -> (note_id, note_date)

        for note in notes:
            note_id = note["note_id"]
            note_date = note.get("date", "")
            text_lower = note.get("text", "").lower()

            # Find stopping phrases, look for med name within 60 chars after
            for match in STOP_PATTERN.finditer(text_lower):
                after = text_lower[match.end():match.end() + 60]
                for med in KNOWN_MEDS:
                    if re.search(r"\b" + med + r"\b", after, re.IGNORECASE):
                        if med not in discontinued:
                            discontinued[med] = (note_id, note_date)

            # Check if a discontinued med reappears as active
            note_entities = next(
                (e for e in extracted_entities if e["note_id"] == note_id), {}
            )
            for med_entity in note_entities.get("entities", {}).get("medications", []):
                if med_entity.get("negated"): continue
                med_text = med_entity["text"].lower().strip()
                for disc_med, (disc_note_id, disc_date) in list(discontinued.items()):
                    if disc_note_id == note_id: continue
                    if disc_med in med_text or med_text in disc_med:
                        d1 = parse_date(disc_date)
                        d2 = parse_date(note_date)
                        months = months_between(d1, d2)
                        med_name = med_entity["text"]
                        flags.append({
                            "flag_type": "medication_copy_forward",
                            "severity": "high",
                            "detail": (
                                med_name + " listed as active in " + note_id
                                + " (" + note_date + "), but patient reported stopping it"
                                + " in " + disc_note_id + " (" + disc_date + ") — "
                                + str(round(months, 1)) + " months earlier."
                                + " No re-initiation documented. Likely copy-forward."
                            ),
                            "note_ids": disc_note_id + " -> " + note_id,
                            "medication": med_name,
                            "discontinued_note": disc_note_id,
                            "reappeared_note": note_id,
                        })
                        del discontinued[disc_med]
                        break

        return flags

    def _check_acute_persistence(self, notes):
        """
        Acute complication still present beyond expected resolution window
        without documented resolution — possible copy-forward.
        """
        flags = []
        first_seen = {}
        last_seen = {}
        resolved = set()

        for note in notes:
            note_id = note["note_id"]
            note_date = note.get("date", "")
            text_lower = note.get("text", "").lower()

            for comp in ACUTE_COMPLICATIONS:
                if comp in text_lower:
                    idx = text_lower.find(comp)
                    context = text_lower[max(0, idx - 80):idx + 80]
                    is_resolved = any(
                        re.search(p, context, re.IGNORECASE)
                        for p in RESOLUTION_PHRASES
                    )
                    if is_resolved:
                        resolved.add(comp)
                    else:
                        if comp not in first_seen:
                            first_seen[comp] = (note_id, note_date)
                        last_seen[comp] = (note_id, note_date)

        for comp, max_days in ACUTE_COMPLICATIONS.items():
            if comp in resolved: continue
            if comp not in first_seen or comp not in last_seen: continue
            first_nid, first_date = first_seen[comp]
            last_nid, last_date = last_seen[comp]
            if first_nid == last_nid: continue
            d1 = parse_date(first_date)
            d2 = parse_date(last_date)
            days = abs((d2 - d1).days) if d1 and d2 else 0
            if days > max_days:
                flags.append({
                    "flag_type": "acute_complication_persistence",
                    "severity": "moderate",
                    "detail": (
                        comp + " first documented in " + first_nid
                        + " (" + first_date + "), still present in "
                        + last_nid + " (" + last_date + ") — "
                        + str(days) + " days. Expected resolution: ~"
                        + str(max_days) + " days. No resolution documented."
                    ),
                    "note_ids": first_nid + " -> " + last_nid,
                    "complication": comp,
                    "days_persistent": days,
                    "expected_max_days": max_days,
                })

        return flags
