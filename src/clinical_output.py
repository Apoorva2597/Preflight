"""
clinical_output.py — Physician-facing summary.
Tabbed interface: Visit View | Chart Reliability | Note Freshness | Entity Detail
"""
from pathlib import Path


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _band(score):
    if score >= 0.80: return "Low risk",     "#0f6e56", "#e1f5ee", "#9fe1cb"
    if score >= 0.60: return "Review items", "#854f0b", "#faeeda", "#ef9f27"
    if score >= 0.40: return "Needs review", "#993c1d", "#faece7", "#d85a30"
    return               "Do not rely",  "#711b13", "#f5e6e6", "#c0392b"


def _ring(score):
    pct  = round(score * 100)
    lbl, clr, bg, acc = _band(score)
    circ = 282.7
    dash = round(circ * score, 1)
    gap  = round(circ - dash, 1)
    return f'''<svg width="110" height="110" viewBox="0 0 110 110" style="display:block">
  <circle cx="55" cy="55" r="40" fill="none" stroke="#e8e8e4" stroke-width="9"/>
  <circle cx="55" cy="55" r="40" fill="none" stroke="{clr}" stroke-width="9"
          stroke-dasharray="{dash} {gap}" stroke-dashoffset="62.8"
          stroke-linecap="round" transform="rotate(-90 55 55)"/>
  <text x="55" y="50" text-anchor="middle" font-size="20" font-weight="800"
        fill="{clr}" font-family="Georgia,serif">{pct}%</text>
  <text x="55" y="65" text-anchor="middle" font-size="10" font-weight="700"
        fill="{clr}" font-family="-apple-system,sans-serif">{lbl}</text>
</svg>'''


def _plain_summary(result):
    """
    Replace 'Probable chart' style labels with plain clinical facts.
    Less words, more impact.
    """
    flags      = result.get("named_flags", [])
    care_gaps  = result.get("care_gaps", [])
    fusion     = result.get("fusion_conflicts", [])
    contra     = result.get("temporal_contradictions", [])
    icd        = result.get("icd_divergence", {})
    freshness  = result.get("freshness", {})

    conflicts  = len([f for f in fusion if f.get("severity") == "high"])
    gaps       = len([g for g in care_gaps if g.get("severity") == "high"])
    uncoded    = len(icd.get("free_text_only", []))
    stagnant   = freshness.get("stagnant_note_count", 0)
    total      = len(result.get("top3_issues", []))

    parts = []
    if conflicts:
        parts.append(f"{conflicts} structured/note conflict{'s' if conflicts>1 else ''}")
    if gaps:
        parts.append(f"{gaps} unmanaged condition{'s' if gaps>1 else ''}")
    if uncoded:
        parts.append(f"{uncoded} uncoded diagnosis{'es' if uncoded>1 else ''}")
    if stagnant:
        parts.append(f"{stagnant} copy-heavy note{'s' if stagnant>1 else ''}")

    if not parts:
        return "No significant documentation issues detected."
    return " · ".join(parts) + " — review before visit."


def _issue_card(rank, issue):
    sev = issue.get("severity", "moderate")
    tc  = {"high": "#993c1d", "moderate": "#854f0b", "low": "#185fa5"}.get(sev, "#555")
    bg  = {"high": "#faece7", "moderate": "#faeeda", "low": "#e6f1fb"}.get(sev, "#f5f5f3")
    bdr = {"high": "#d85a30", "moderate": "#ef9f27", "low": "#378add"}.get(sev, "#ccc")
    rc  = {1: "#c0392b", 2: "#b06000", 3: "#185fa5"}.get(rank, "#555")

    evidence_items = ""
    for e in issue.get("evidence", []):
        # Bold the note reference if present
        import re
        e_fmt = re.sub(r'(note_\d+)', r'<strong>\1</strong>', e)
        evidence_items += f'<li style="margin-bottom:5px">{e_fmt}</li>'

    ct_label = issue.get("conflict_type","").replace("_"," ").title()

    return f'''
    <div style="border:1.5px solid {bdr};border-radius:10px;
                margin-bottom:14px;overflow:hidden;background:white">
      <div style="background:{bg};padding:11px 16px;
                  display:flex;align-items:center;gap:12px;
                  border-bottom:1px solid {bdr}44">
        <div style="width:24px;height:24px;border-radius:50%;
                    background:{rc};display:flex;align-items:center;
                    justify-content:center;flex-shrink:0;
                    font-size:12px;font-weight:800;color:white">{rank}</div>
        <div style="flex:1">
          <div style="font-size:10px;font-weight:800;color:{tc};
                      text-transform:uppercase;letter-spacing:.07em;
                      margin-bottom:2px">{ct_label}</div>
          <div style="font-size:14px;font-weight:700;color:#1a1a1a;
                      line-height:1.35">{issue.get("problem","")}</div>
        </div>
        <span style="background:{tc};color:white;font-size:10px;
                     font-weight:700;padding:2px 8px;border-radius:8px;
                     flex-shrink:0;text-transform:uppercase">{sev}</span>
      </div>
      <div style="padding:14px 16px;display:grid;
                  grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                      letter-spacing:.07em;color:#9b9b94;margin-bottom:7px">
            Evidence</div>
          <ul style="font-size:12px;line-height:1.65;
                     padding-left:14px;color:#2a2a24;margin:0">
            {evidence_items}
          </ul>
        </div>
        <div>
          <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                      letter-spacing:.07em;color:#9b9b94;margin-bottom:7px">
            Action</div>
          <div style="font-size:12px;color:#1a1a1a;line-height:1.65;
                      padding:8px 10px;background:#f0faf5;border-radius:6px;
                      border-left:3px solid #0f6e56">
            {issue.get("action","")}</div>
        </div>
      </div>
    </div>'''


def generate_clinical_html(result, path):
    pid       = result["patient_id"]
    trust     = result.get("trust_score", 0)
    top3      = result.get("top3_issues", [])
    scored    = result.get("scored_entities", [])
    freshness = result.get("freshness", {})
    icd       = result.get("icd_divergence", {})

    lbl, clr, bg, acc = _band(trust)
    ring    = _ring(trust)
    summary = _plain_summary(result)
    fscore  = freshness.get("record_freshness_score", 1.0)
    fc      = "#0f6e56" if fscore >= 0.60 else "#854f0b" if fscore >= 0.35 else "#993c1d"

    # ── Tab 1: Visit View — Top 3 ────────────────────────────────────────────
    if top3:
        issues_html = "".join(_issue_card(i+1, issue) for i, issue in enumerate(top3))
    else:
        issues_html = '<p style="color:#9b9b94;font-size:14px">No significant issues detected.</p>'

    # ── Tab 2: Chart Reliability ─────────────────────────────────────────────
    score_meaning = {
        "Low risk":     "Documentation is consistent across notes and structured data. Standard verification applies.",
        "Review items": "Chart has items worth verifying. Check flagged issues before coding or clinical decisions.",
        "Needs review": "Multiple documentation gaps or conflicts detected. Review all flagged issues before this visit.",
        "Do not rely":  "Significant conflicts detected. Do not rely on this chart without manual review.",
    }.get(lbl, "")

    icd_rows = ""
    for u in icd.get("free_text_only", []):
        icd_rows += f'''<div style="padding:8px 12px;background:#fff8e6;border-radius:6px;
                            margin-bottom:6px;font-size:12px;color:#854f0b;
                            border-left:3px solid #ef9f27">
          <strong>{u.get("diagnosis","").title()}</strong> — in notes, not coded.
          Risk: undercoding, invisible to ICD-based queries.</div>'''
    for c in icd.get("code_only", []):
        icd_rows += f'''<div style="padding:8px 12px;background:#faece7;border-radius:6px;
                            margin-bottom:6px;font-size:12px;color:#993c1d;
                            border-left:3px solid #d85a30">
          <strong>{c.get("icd_prefix","")}</strong> — coded but no narrative support found.
          Risk: may not survive audit.</div>'''
    if not icd_rows:
        icd_rows = '<div style="font-size:12px;color:#9b9b94">No ICD divergence detected.</div>'

    reliability_tab = f'''
      <div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:20px">
        <div style="flex-shrink:0">{ring}</div>
        <div>
          <div style="font-size:18px;font-weight:700;color:{clr};margin-bottom:4px">{lbl}</div>
          <div style="font-size:13px;color:#444;line-height:1.6;max-width:480px;
                      margin-bottom:12px">{score_meaning}</div>
          <div style="font-size:12px;color:#6b6b65">
            How to read this score: above 80% means documentation is internally consistent.
            60–80% means review flagged items. Below 60% means do not rely without manual review.
          </div>
        </div>
      </div>
      <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                  letter-spacing:.08em;color:#9b9b94;margin-bottom:10px">
        Source hierarchy used for scoring</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:20px">
        <span style="font-size:11px;font-weight:700;color:#0f6e56;
               background:#eaf7f0;padding:3px 10px;border-radius:10px">
          Tier 1: Lab / pathology (highest)</span>
        <span style="font-size:11px;font-weight:700;color:#2563a8;
               background:#e8f0fb;padding:3px 10px;border-radius:10px">
          Tier 2: Medication orders</span>
        <span style="font-size:11px;font-weight:700;color:#b06000;
               background:#fff8e6;padding:3px 10px;border-radius:10px">
          Tier 3: Problem list</span>
        <span style="font-size:11px;font-weight:700;color:#555;
               background:#f5f5f3;padding:3px 10px;border-radius:10px">
          Tier 4: Recent notes</span>
        <span style="font-size:11px;font-weight:700;color:#993c1d;
               background:#faece7;padding:3px 10px;border-radius:10px">
          Tier 5: Old notes (copy-forward risk)</span>
      </div>
      <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                  letter-spacing:.08em;color:#9b9b94;margin-bottom:10px">
        ICD divergence</div>
      {icd_rows}'''

    # ── Tab 3: Note Freshness ────────────────────────────────────────────────
    notes_data = freshness.get("notes", [])
    bars = ""
    for n in notes_data:
        pct   = int(n["freshness_score"] * 100)
        color = n["freshness_color"]
        nid   = n["note_id"].replace("note_", "Note ")
        date  = n.get("date", "")
        label = n["freshness_label"]
        new_c = n.get("new_sentence_count", 0)
        cop_c = n.get("copied_sentence_count", 0)
        bars += f'''
        <div style="margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="font-size:12px;font-weight:700;color:#1a1a1a;
                         min-width:60px">{nid}</span>
            <span style="font-size:11px;color:#9b9b94;min-width:80px">{date}</span>
            <div style="flex:1;background:#eeede8;border-radius:4px;
                        height:14px;overflow:hidden">
              <div style="height:100%;width:{pct}%;background:{color};
                          border-radius:4px;min-width:3px"></div>
            </div>
            <span style="font-size:12px;font-weight:800;color:{color};
                         min-width:36px;text-align:right">{pct}%</span>
            <span style="font-size:11px;font-weight:700;color:{color};
                         min-width:64px">{label}</span>
          </div>
          <div style="font-size:11px;color:#9b9b94;padding-left:150px">
            {new_c} new sentences · {cop_c} copied from prior note</div>
        </div>'''

    stagnant_notes = [n for n in notes_data if n["freshness_label"] in ("Stagnant","Copy")]
    freshness_tab = f'''
      <div style="font-size:13px;color:#444;margin-bottom:16px">
        Each bar shows what percentage of a note contains new clinical content
        vs. text carried forward from the prior note.
        <strong style="color:{fc}">Record freshness: {round(fscore*100)}%</strong> —
        {freshness.get("fresh_note_count",0)} of {freshness.get("total_notes",0)} notes
        contain substantial new content.
        {f'<strong style="color:#993c1d">{len(stagnant_notes)} note(s) are predominantly copy-forwarded.</strong>' if stagnant_notes else ''}
      </div>
      {bars if bars else '<p style="color:#9b9b94;font-size:13px">No notes available.</p>'}'''

    # ── Tab 4: Entity Detail ─────────────────────────────────────────────────
    EXCLUDE = {"structural_anatomic"}
    low_conf = [
        e for e in scored
        if e.get("composite_score", 1) < 0.70
        and e.get("condition_category", "") not in EXCLUDE
    ]

    grouped = {}
    for e in low_conf:
        cat = e.get("condition_category","other").replace("_"," ").title()
        grouped.setdefault(cat, []).append(e)

    entity_rows = ""
    for cat, entities in grouped.items():
        entity_rows += f'''<div style="font-size:10px;font-weight:800;
                               text-transform:uppercase;letter-spacing:.07em;
                               color:#9b9b94;margin:14px 0 7px">{cat}</div>'''
        for e in entities[:8]:
            score = e.get("composite_score", 0)
            el, ec, ebg, _ = _band(score)
            pct  = int(score * 100)
            rec  = e.get("recommendation","").split(".")[0]
            apps = e.get("appearances", 1)
            entity_rows += f'''
            <div style="display:flex;align-items:center;gap:10px;
                        padding:8px 12px;background:{ebg};border-radius:8px;
                        margin-bottom:5px">
              <div style="font-size:13px;font-weight:700;
                          color:#1a1a1a;min-width:150px">
                {e.get("entity","").title()}</div>
              <div style="flex:1;height:7px;background:#00000012;
                          border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{pct}%;
                            background:{ec};border-radius:3px"></div>
              </div>
              <div style="font-size:12px;font-weight:800;
                          color:{ec};min-width:36px;text-align:right">{pct}%</div>
              <span style="font-size:10px;font-weight:700;color:{ec};
                           border:1.5px solid {ec};padding:2px 7px;
                           border-radius:8px;min-width:80px;text-align:center">{el}</span>
              <div style="font-size:11px;color:#6b6b65;
                          min-width:50px;text-align:center">{apps}x noted</div>
              <div style="font-size:11px;color:#6b6b65;
                          max-width:240px;line-height:1.4">{rec}</div>
            </div>'''

    if not entity_rows:
        entity_rows = '<p style="font-size:13px;color:#9b9b94">All entities above confidence threshold.</p>'

    entity_tab = f'''
      <div style="font-size:13px;color:#444;margin-bottom:14px">
        Entities scored below 70% confidence. Score reflects source tier,
        time since last verification, and corroboration across notes.
        Structural facts (mastectomy, etc.) are excluded — low scores
        on permanent facts reflect documentation patterns, not clinical uncertainty.
      </div>
      {entity_rows}'''

    # ── Full HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chart Summary — {pid}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#f5f4ef;color:#1a1a1a;line-height:1.6}}
  .page{{max-width:880px;margin:0 auto;padding:1.75rem 1.5rem 3rem}}
  .header{{margin-bottom:1.5rem;padding-bottom:1rem;
           border-bottom:2px solid #1a1a1a}}
  h1{{font-family:Georgia,serif;font-size:24px;font-weight:400}}
  .meta{{font-size:11px;color:#9b9b94;margin-top:3px}}
  .summary-bar{{background:white;border-radius:10px;padding:12px 18px;
                margin-bottom:1.25rem;font-size:13px;font-weight:600;
                color:#2a2a24;border-left:4px solid {clr};
                box-shadow:0 1px 3px rgba(0,0,0,.06)}}
  /* Tabs */
  .tabs{{display:flex;gap:0;margin-bottom:0;border-bottom:2px solid #e0dfd8}}
  .tab{{padding:9px 18px;font-size:12px;font-weight:700;cursor:pointer;
        color:#9b9b94;border-bottom:2px solid transparent;
        margin-bottom:-2px;transition:all .15s;user-select:none;
        background:none;border-top:none;border-left:none;border-right:none}}
  .tab:hover{{color:#1a1a1a}}
  .tab.active{{color:#1a1a1a;border-bottom:2px solid #1a1a1a}}
  .tab-content{{background:white;border-radius:0 10px 10px 10px;
                padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.06),
                0 0 0 1px rgba(0,0,0,.04);display:none}}
  .tab-content.active{{display:block}}
  .disclaimer{{font-size:11px;color:#b0afa8;margin-top:1.5rem;
               padding-top:1rem;border-top:1px solid #e0dfd8;
               text-align:center}}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <h1>Chart Summary — {pid}</h1>
    <div class="meta">EHR Temporal Validator &nbsp;&middot;&nbsp;
      github.com/Apoorva2597/ehr-temporal-validator &nbsp;&middot;&nbsp;
      Full technical report available</div>
  </div>

  <div class="summary-bar">{summary}</div>

  <div class="tabs">
    <button class="tab active" onclick="showTab('visit',this)">
      Visit View</button>
    <button class="tab" onclick="showTab('reliability',this)">
      Chart Reliability</button>
    <button class="tab" onclick="showTab('freshness',this)">
      Note Freshness</button>
    <button class="tab" onclick="showTab('entities',this)">
      Entity Detail</button>
  </div>

  <div id="tab-visit" class="tab-content active">
    <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                letter-spacing:.08em;color:#9b9b94;margin-bottom:14px">
      Top {len(top3)} issues — review before visit</div>
    {issues_html}
  </div>

  <div id="tab-reliability" class="tab-content">
    {reliability_tab}
  </div>

  <div id="tab-freshness" class="tab-content">
    {freshness_tab}
  </div>

  <div id="tab-entities" class="tab-content">
    {entity_tab}
  </div>

  <div class="disclaimer">
    Decision-support tool only. Clinical judgment supersedes all automated flags.
    &nbsp;&middot;&nbsp;
    Apoorva Kolhatkar &nbsp;&middot;&nbsp; MHI, University of Michigan (May 2026)
    &nbsp;&middot;&nbsp; Michigan Medicine NLP Research
  </div>

</div>

<script>
function showTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Clinical HTML saved -> {path}")
