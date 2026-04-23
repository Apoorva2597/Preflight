"""
timeline_output.py
Visual dashboard with clear contrast, readable typography, named flags.
"""
import json
from pathlib import Path


class TimelineOutput:

    def save_json(self, result, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  JSON saved -> {path}")

    def save_html(self, result, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._render(result))
        print(f"  HTML saved -> {path}")

    def _score_color(self, s):
        if s >= 0.80: return "#1a7f5a"
        if s >= 0.60: return "#b06000"
        if s >= 0.40: return "#c0392b"
        return "#7b1c1c"

    def _score_bg(self, s):
        if s >= 0.80: return "#eaf7f0"
        if s >= 0.60: return "#fff8e6"
        if s >= 0.40: return "#fdecea"
        return "#f5e6e6"

    def _label_style(self, label):
        return {
            "confirmed":   ("#1a7f5a","#eaf7f0"),
            "probable":    ("#2563a8","#e8f0fb"),
            "provisional": ("#b06000","#fff8e6"),
            "uncertain":   ("#c0392b","#fdecea"),
            "needs_review": ("#7b1c1c","#f5e6e6"),
        }.get(label, ("#555","#f5f5f3"))

    def _cf_badge(self, cf):
        if cf >= 0.60: return ("#c0392b","#fdecea","High")
        if cf >= 0.30: return ("#b06000","#fff8e6","Moderate")
        return ("#1a7f5a","#eaf7f0","Low")

    def _render(self, r):
        pid   = r["patient_id"]
        trust = r.get("trust_score", 0)
        scored= r.get("scored_entities", [])
        conts = r.get("temporal_contradictions", [])
        icd   = r.get("icd_divergence", {})
        tl    = r.get("timeline", [])
        flags = r.get("named_flags", [])

        tc = self._score_color(trust)
        tl_label = ("Trusted" if trust>=0.80 else "Probable" if trust>=0.60
                    else "Provisional" if trust>=0.40 else "Unreliable")

        # stat cards
        low_conf = len([e for e in scored if e.get("composite_score",1)<0.40])
        uncoded  = len(icd.get("free_text_only",[]))

        # named flags
        flag_html = ""
        if flags:
            rows = ""
            for f in flags:
                sc_ = {"high":"#c0392b","moderate":"#b06000","low":"#2563a8"}.get(f.get("severity","moderate"),"#555")
                sb_ = {"high":"#fdecea","moderate":"#fff8e6","low":"#e8f0fb"}.get(f.get("severity","moderate"),"#f5f5f3")
                rows += f'''
                <div style="border-left:3px solid {sc_};padding:10px 14px;margin-bottom:8px;
                            background:#fafaf8;border-radius:0 6px 6px 0">
                  <div style="font-size:12px;font-weight:700;color:{sc_};margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em">
                    {f.get("flag_type","").replace("_"," ")}</div>
                  <div style="font-size:13px;color:#1a1a1a;margin-bottom:3px">{f.get("detail","")}</div>
                  <div style="font-size:11px;color:#aaa">{f.get("note_ids","")}</div>
                </div>'''
            flag_html = f'''
            <div class="card">
              <div class="card-title">Named flags ({len(flags)})</div>
              {rows}
            </div>'''

        # timeline rows
        type_style = {
            "procedure":          ("#2563a8","#e8f0fb"),
            "complication":       ("#c0392b","#fdecea"),
            "diagnosis":          ("#b06000","#fff8e6"),
            "admission_discharge":("#1a7f5a","#eaf7f0"),
        }
        tl_rows = ""
        for e in tl[:25]:
            et = e.get("event_type","")
            ec,eb = type_style.get(et,("#555","#f5f5f3"))
            tl_rows += f'''
            <tr>
              <td style="padding:8px 12px;font-size:12px;color:#666;white-space:nowrap">{e.get("date","—")}</td>
              <td style="padding:8px 12px">
                <span style="background:{eb};color:{ec};font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px;white-space:nowrap">{et}</span></td>
              <td style="padding:8px 12px;font-size:13px;color:#1a1a1a;max-width:400px">{e.get("text","")[:130]}</td>
              <td style="padding:8px 12px;font-size:11px;color:#bbb">{e.get("note_id","")}</td>
            </tr>'''

        # scored entity rows
        sc_rows = ""
        for e in scored[:20]:
            score = e.get("composite_score",0)
            ec = self._score_color(score)
            eb = self._score_bg(score)
            label = e.get("classification","")
            lc,lb = self._label_style(label)
            cf = e.get("copy_forward_suspicion",0)
            cfc,cfb,cfl = self._cf_badge(cf)
            sc_rows += f'''
            <tr>
              <td style="padding:8px 12px;font-size:13px;font-weight:600;color:#1a1a1a">{e.get("entity","")}</td>
              <td style="padding:8px 12px;font-size:12px;color:#666">{e.get("condition_category","").replace("_"," ")}</td>
              <td style="padding:8px 12px">
                <span style="background:{eb};color:{ec};font-size:13px;font-weight:700;
                             padding:4px 10px;border-radius:6px">{score}</span></td>
              <td style="padding:8px 12px">
                <span style="background:{lb};color:{lc};font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px">{label}</span></td>
              <td style="padding:8px 12px">
                <span style="background:{cfb};color:{cfc};font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px">{cfl} ({cf})</span></td>
              <td style="padding:8px 12px;font-size:11px;color:#888;max-width:240px">{e.get("recommendation","")[:100]}</td>
            </tr>'''

        # contradiction rows
        cont_rows = ""
        for c in conts:
            sev = c.get("severity","moderate")
            sc_ = {"high":"#c0392b","moderate":"#b06000","low":"#2563a8"}.get(sev,"#555")
            sb_ = {"high":"#fdecea","moderate":"#fff8e6","low":"#e8f0fb"}.get(sev,"#f5f5f3")
            cont_rows += f'''
            <tr>
              <td style="padding:8px 12px">
                <span style="background:{sb_};color:{sc_};font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px">{sev}</span></td>
              <td style="padding:8px 12px;font-size:12px;color:#666">{c.get("contradiction_type","").replace("_"," ")}</td>
              <td style="padding:8px 12px;font-size:13px;color:#1a1a1a">{c.get("note","")}</td>
            </tr>'''

        # icd rows
        icd_rows = ""
        for i in icd.get("free_text_only",[]):
            icd_rows += f'''
            <tr>
              <td style="padding:8px 12px;font-size:13px;font-weight:600;color:#1a1a1a">{i.get("diagnosis","")}</td>
              <td style="padding:8px 12px">
                <span style="background:#fff8e6;color:#b06000;font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px">Free text only</span></td>
              <td style="padding:8px 12px;font-size:12px;color:#888">{i.get("risk","")}</td>
            </tr>'''
        for i in icd.get("code_only",[]):
            icd_rows += f'''
            <tr>
              <td style="padding:8px 12px;font-size:13px;font-weight:600;color:#1a1a1a">
                {i.get("icd_prefix","")} — {", ".join(i.get("expected_diagnoses",[])[:2])}</td>
              <td style="padding:8px 12px">
                <span style="background:#fdecea;color:#c0392b;font-size:11px;font-weight:700;
                             padding:3px 9px;border-radius:12px">Code only</span></td>
              <td style="padding:8px 12px;font-size:12px;color:#888">{i.get("risk","")}</td>
            </tr>'''

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EHR Temporal Validator — {pid}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#f0f0ef;color:#1a1a1a;padding:2rem;line-height:1.6}}
  .header{{margin-bottom:1.5rem}}
  .header h1{{font-size:22px;font-weight:800;letter-spacing:-.4px}}
  .header .sub{{font-size:12px;color:#999;margin-top:4px}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem}}
  .stat{{background:white;border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .stat-num{{font-size:28px;font-weight:800}}
  .stat-lbl{{font-size:11px;color:#999;margin-top:2px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
  .stat-sub{{font-size:11px;color:#bbb;margin-top:5px}}
  .bar{{height:6px;background:#eee;border-radius:3px;margin-top:8px;overflow:hidden}}
  .fill{{height:100%;border-radius:3px;background:{tc};width:{int(trust*100)}%}}
  .card{{background:white;border-radius:10px;padding:20px 22px;
         margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .card-title{{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
               color:#aaa;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #f0f0ef}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;font-size:10px;color:#bbb;font-weight:700;
      padding:6px 12px;border-bottom:1px solid #eee;
      text-transform:uppercase;letter-spacing:.06em}}
  tr:hover td{{background:#fafaf8}}
  td{{border-bottom:1px solid #f5f5f3;vertical-align:middle}}
  .empty{{font-size:13px;color:#ccc;padding:8px 0}}
  .footer{{font-size:11px;color:#ccc;margin-top:2rem;text-align:center;padding-top:1rem;
           border-top:1px solid #e8e8e8}}
</style>
</head>
<body>

<div class="header">
  <h1>EHR Temporal Validator</h1>
  <div class="sub">Patient {pid} &nbsp;&middot;&nbsp; github.com/Apoorva2597/ehr-temporal-validator &nbsp;&middot;&nbsp; Apoorva Kolhatkar, MHI &mdash; University of Michigan</div>
</div>

<div class="grid">
  <div class="stat">
    <div class="stat-num" style="color:{tc}">{trust}</div>
    <div class="stat-lbl">Trust score</div>
    <div class="bar"><div class="fill"></div></div>
    <div style="font-size:11px;color:{tc};margin-top:5px;font-weight:700">{tl_label}</div>
  </div>
  <div class="stat">
    <div class="stat-num">{len(scored)}</div>
    <div class="stat-lbl">Entities scored</div>
    <div class="stat-sub">{low_conf} below 0.40 threshold</div>
  </div>
  <div class="stat">
    <div class="stat-num" style="color:{'#c0392b' if conts else '#1a7f5a'}">{len(conts)}</div>
    <div class="stat-lbl">Temporal contradictions</div>
    <div class="stat-sub">Ordering violations</div>
  </div>
  <div class="stat">
    <div class="stat-num" style="color:{'#b06000' if uncoded else '#1a7f5a'}">{uncoded}</div>
    <div class="stat-lbl">Uncoded diagnoses</div>
    <div class="stat-sub">{icd.get("summary","")}</div>
  </div>
</div>

{flag_html}

<div class="card">
  <div class="card-title">Clinical timeline &mdash; {len(tl)} events extracted</div>
  <table>
    <tr><th>Date</th><th>Type</th><th>Text</th><th>Note</th></tr>
    {tl_rows or '<tr><td colspan="4" class="empty" style="padding:12px">No events extracted.</td></tr>'}
  </table>
</div>

<div class="card">
  <div class="card-title">Entity confidence scores &mdash; lowest confidence first</div>
  <table>
    <tr><th>Entity</th><th>Category</th><th>Score</th><th>Classification</th><th>CF Suspicion</th><th>Recommendation</th></tr>
    {sc_rows or '<tr><td colspan="6" class="empty" style="padding:12px">No entities scored.</td></tr>'}
  </table>
  <div style="font-size:11px;color:#ccc;margin-top:10px">CF = copy-forward suspicion, weighted by condition-category decay rate (λ)</div>
</div>

<div class="card">
  <div class="card-title">Temporal contradictions &mdash; {len(conts)} detected</div>
  {'<table><tr><th>Severity</th><th>Type</th><th>Detail</th></tr>' + cont_rows + '</table>'
    if cont_rows else '<p class="empty">No temporal contradictions detected.</p>'}
</div>

<div class="card">
  <div class="card-title">ICD divergence &mdash; {icd.get("summary","no data")}</div>
  {'<table><tr><th>Diagnosis / Code</th><th>Source</th><th>Risk</th></tr>' + icd_rows + '</table>'
    if icd_rows else '<p class="empty">No ICD divergence detected.</p>'}
</div>

<div class="footer">
  Apoorva Kolhatkar &nbsp;&middot;&nbsp; MHI Candidate, University of Michigan (May 2026) &nbsp;&middot;&nbsp;
  Michigan Medicine NLP Research &nbsp;&middot;&nbsp;
  Prototype validated on synthetic data &mdash; MIMIC-III integration supported with credentialed PhysioNet access
</div>
</body>
</html>"""
