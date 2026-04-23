"""
compare.py
--compare mode: runs naive (trust-everything) vs. validated pipeline
side by side and generates a comparison HTML report.

Shows concretely what a downstream system would do differently
if it relied on the raw chart vs. the trust-scored chart.
"""

import json
from pathlib import Path


def generate_compare_html(naive: dict, validated: dict, path: str):
    """
    Generate a side-by-side comparison HTML for one patient.
    naive     = result from treating all entities as fully reliable
    validated = result from composite confidence scoring pipeline
    """
    pid = validated["patient_id"]

    # What the naive system "sees" as reliable
    naive_meds  = naive.get("active_medications", [])
    naive_diags = naive.get("active_diagnoses", [])
    naive_icd   = naive.get("assumed_icd_codes", [])

    # What the validated system flags
    flags  = validated.get("named_flags", [])
    scored = validated.get("scored_entities", [])
    icd    = validated.get("icd_divergence", {})
    trust  = validated.get("trust_score", 0)

    low_conf = [e for e in scored if e.get("composite_score", 1) < 0.40]
    high_cf  = [e for e in scored if e.get("copy_forward_suspicion", 0) > 0.60]
    uncoded  = icd.get("free_text_only", [])

    # Build naive entity rows
    def naive_row(entity, etype):
        return f"""
        <tr>
          <td style="padding:8px 12px;font-size:13px;font-weight:500;color:#1a1a1a">{entity}</td>
          <td style="padding:8px 12px">
            <span style="background:#eaf7f0;color:#1a7f5a;font-size:11px;font-weight:700;
                         padding:3px 9px;border-radius:12px">Accepted</span></td>
          <td style="padding:8px 12px;font-size:12px;color:#aaa">{etype} — no validation applied</td>
        </tr>"""

    # Build validation catch rows
    def catch_row(entity, issue, severity):
        sc_ = {"high":"#c0392b","moderate":"#b06000","low":"#2563a8"}.get(severity,"#555")
        sb_ = {"high":"#fdecea","moderate":"#fff8e6","low":"#e8f0fb"}.get(severity,"#f5f5f3")
        return f"""
        <tr>
          <td style="padding:8px 12px;font-size:13px;font-weight:500;color:#1a1a1a">{entity}</td>
          <td style="padding:8px 12px">
            <span style="background:{sb_};color:{sc_};font-size:11px;font-weight:700;
                         padding:3px 9px;border-radius:12px">Flagged</span></td>
          <td style="padding:8px 12px;font-size:12px;color:#555">{issue}</td>
        </tr>"""

    naive_rows = "".join(naive_row(e, "medication") for e in naive_meds[:8])
    naive_rows += "".join(naive_row(e, "diagnosis") for e in naive_diags[:8])
    if not naive_rows:
        naive_rows = '<tr><td colspan="3" style="padding:12px;color:#ccc;font-size:13px">No entities — run pipeline first.</td></tr>'

    catch_rows = ""
    for f in flags:
        catch_rows += catch_row(
            f.get("medication", f.get("complication", "entity")),
            f.get("detail", "")[:120],
            f.get("severity", "moderate")
        )
    for e in low_conf[:5]:
        catch_rows += catch_row(
            e.get("entity",""),
            f"Composite score {e.get('composite_score',0)} — {e.get('classification','')}. {e.get('recommendation','')[:80]}",
            "moderate"
        )
    for u in uncoded[:3]:
        catch_rows += catch_row(
            u.get("diagnosis",""),
            f"In narrative text but never coded — invisible to ICD-based queries.",
            "moderate"
        )
    if not catch_rows:
        catch_rows = '<tr><td colspan="3" style="padding:12px;color:#ccc;font-size:13px">No issues detected.</td></tr>'

    tc = "#1a7f5a" if trust >= 0.80 else "#b06000" if trust >= 0.60 else "#c0392b"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EHR Validator — Compare Mode — {pid}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#f0f0ef;color:#1a1a1a;padding:2rem;line-height:1.6}}
  h1{{font-size:22px;font-weight:800;letter-spacing:-.4px;margin-bottom:4px}}
  .sub{{font-size:12px;color:#999;margin-bottom:1.5rem}}
  .compare-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:1.5rem}}
  .panel{{background:white;border-radius:10px;padding:20px 22px;
          box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .panel-title{{font-size:11px;font-weight:800;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:14px;padding-bottom:10px;
                border-bottom:2px solid currentColor}}
  .naive-title{{color:#2563a8}}
  .val-title{{color:#1a7f5a}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;font-size:10px;color:#bbb;font-weight:700;
      padding:6px 12px;border-bottom:1px solid #eee;
      text-transform:uppercase;letter-spacing:.06em}}
  tr:hover td{{background:#fafaf8}}
  td{{border-bottom:1px solid #f5f5f3;vertical-align:middle}}
  .card{{background:white;border-radius:10px;padding:20px 22px;
         box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:1.25rem}}
  .card-title{{font-size:11px;font-weight:800;text-transform:uppercase;
               letter-spacing:.08em;color:#aaa;margin-bottom:14px;
               padding-bottom:10px;border-bottom:1px solid #f0f0ef}}
  .summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:1.5rem}}
  .stat{{background:white;border-radius:10px;padding:16px 18px;
         box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .stat-num{{font-size:26px;font-weight:800}}
  .stat-lbl{{font-size:11px;color:#999;font-weight:600;text-transform:uppercase;
             letter-spacing:.04em;margin-top:2px}}
  .stat-sub{{font-size:11px;color:#bbb;margin-top:5px}}
  .footer{{font-size:11px;color:#ccc;margin-top:2rem;text-align:center;
           padding-top:1rem;border-top:1px solid #e8e8e8}}
</style>
</head>
<body>

<h1>EHR Temporal Validator — Compare Mode</h1>
<div class="sub">Patient {pid} &nbsp;&middot;&nbsp; Naive (trust-everything) vs. Validated (composite confidence scoring) &nbsp;&middot;&nbsp; github.com/Apoorva2597/ehr-temporal-validator</div>

<div class="summary-grid">
  <div class="stat">
    <div class="stat-num" style="color:{tc}">{trust}</div>
    <div class="stat-lbl">Chart trust score</div>
    <div class="stat-sub">After validation pipeline</div>
  </div>
  <div class="stat">
    <div class="stat-num" style="color:#c0392b">{len(flags)}</div>
    <div class="stat-lbl">Named flags</div>
    <div class="stat-sub">Invisible to naive system</div>
  </div>
  <div class="stat">
    <div class="stat-num" style="color:#b06000">{len(uncoded)}</div>
    <div class="stat-lbl">Uncoded diagnoses</div>
    <div class="stat-sub">In free text, not ICD fields</div>
  </div>
</div>

<div class="compare-grid">
  <div class="panel">
    <div class="panel-title naive-title">Naive system &mdash; trusts the chart</div>
    <p style="font-size:12px;color:#888;margin-bottom:12px">
      Treats all documented entities as reliable. No temporal validation,
      no copy-forward detection, no ICD divergence check.
      This is what most EHR summarization systems do today.
    </p>
    <table>
      <tr><th>Entity</th><th>Status</th><th>Basis</th></tr>
      {naive_rows}
    </table>
  </div>

  <div class="panel">
    <div class="panel-title val-title">Validated system &mdash; scores the chart</div>
    <p style="font-size:12px;color:#888;margin-bottom:12px">
      Applies composite confidence scoring: source reliability &times;
      temporal decay &times; corroboration &times; (1 &minus; contradiction penalty).
      Condition-specific decay rates. Named flags for clinically significant errors.
    </p>
    <table>
      <tr><th>Entity</th><th>Status</th><th>Issue detected</th></tr>
      {catch_rows}
    </table>
  </div>
</div>

<div class="card">
  <div class="card-title">What changes downstream</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <div style="font-size:12px;font-weight:700;color:#2563a8;margin-bottom:8px">Naive system would:</div>
      <ul style="font-size:13px;color:#555;padding-left:16px;line-height:2">
        <li>List discontinued medications as active in patient summary</li>
        <li>Miss free-text diagnoses in ICD-based coding queries</li>
        <li>Treat copy-forwarded chronic conditions identically to recently verified ones</li>
        <li>Generate coding suggestions based on a potentially corrupted record</li>
      </ul>
    </div>
    <div>
      <div style="font-size:12px;font-weight:700;color:#1a7f5a;margin-bottom:8px">Validated system surfaces:</div>
      <ul style="font-size:13px;color:#555;padding-left:16px;line-height:2">
        <li>Medications documented as active after patient reported stopping them</li>
        <li>Diagnoses in narrative text that were never coded</li>
        <li>Entities with low composite confidence flagged for clinical review</li>
        <li>Temporal contradictions in procedure &rarr; complication ordering</li>
      </ul>
    </div>
  </div>
  <div style="margin-top:14px;padding:12px 14px;background:#f8f8f6;border-radius:8px;
              font-size:12px;color:#888;font-style:italic">
    The pipeline does not make clinical decisions. It reduces the haystack —
    surfacing cases that warrant human review before downstream AI systems
    treat the chart as ground truth.
  </div>
</div>

<div class="footer">
  Apoorva Kolhatkar &nbsp;&middot;&nbsp; MHI Candidate, University of Michigan (May 2026) &nbsp;&middot;&nbsp;
  Michigan Medicine NLP Research &nbsp;&middot;&nbsp;
  Prototype validated on synthetic data
</div>
</body>
</html>"""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Compare HTML saved -> {path}")
