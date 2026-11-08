"""Build the E1 hand-writing workbook (PDF) and the typing template.

    python E1/make_workbook.py select     # pick 75 real arXiv papers posted 2025-01..2025-10 (seed 0)
    python E1/make_workbook.py extract    # question A for each, with the ONE fixed prompt EXTRACT_RQ
    python E1/make_workbook.py pdf        # E1/workbook.pdf + E1/contrast_template.jsonl

Only A comes from the language model, as Task 1 allows ("extract each question with one fixed prompt and
edit it by hand"). Every other field (B, B_min, C, C_tilde, narrower, broader) is left empty to be written by hand.
"""
import json
import random
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import read_jsonl, write_jsonl  # noqa: E402

HERE = Path(__file__).parent
TYPES = ["role_swap", "constraint_flip", "goal_swap", "inquiry_change", "direction_flip"]
N = 75
TYPE_HELP = {
    "role_swap": 'swap two roles: "how does X affect Y" &rarr; "how does Y affect X"',
    "constraint_flip": 'flip a constraint: "without labels" &rarr; "with labels", "on CPUs" &rarr; "on GPUs"',
    "goal_swap": 'change the goal: accuracy &rarr; speed, robustness &rarr; fairness',
    "inquiry_change": 'change the kind of question: "how does" &rarr; "why does" / "can we" / "to what extent"',
    "direction_flip": 'flip the direction: "improve" &rarr; "harm", "increase" &rarr; "reduce"',
}


def cmd_select():
    from common.data_sources import load_arxiv
    P = [p for p in load_arxiv(("cs.LG", "cs.CL", "cs.CV"), "2025-01-01", "2025-10-31")
         if 80 <= len(p["abstract"].split()) <= 220]
    rng = random.Random(0)
    pick = rng.sample(P, N)
    rows = []
    for i, p in enumerate(pick):
        rows.append({"id": f"h{i + 1:03d}", "source_arxiv": p["id"], "date": p["date"], "title": p["title"],
                     "abstract": p["abstract"], "C_type": TYPES[i % 5], "make_broader_longer": i % 5 == i // 5 % 5,
                     "A_extracted": None})
    write_jsonl(HERE / "workbook_items.jsonl", rows)
    print(len(rows), "papers selected")


def cmd_extract():
    from common.extract import extract_rq, extractor
    llm = extractor()
    rows = read_jsonl(HERE / "workbook_items.jsonl")
    for r in rows:
        if not r["A_extracted"]:
            r["A_extracted"] = extract_rq(llm, r["title"], r["abstract"])
            write_jsonl(HERE / "workbook_items.jsonl", rows)  # resumable
    print("extracted", sum(bool(r["A_extracted"]) for r in rows))


CSS = """
@page { size: A4; margin: 14mm 14mm; }
body { font-family: Georgia, serif; font-size: 10pt; line-height: 1.35; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6px; } h2 { font-size: 13pt; color: #7a0c2e; margin: 14px 0 6px; }
.item { break-inside: avoid; break-after: page; }
.meta { font-size: 8.5pt; color: #555; } .abs { font-size: 8.8pt; color: #333; margin: 4px 0 6px; }
.A { background: #eef4fb; border-left: 3px solid #2a5ea8; padding: 4px 6px; margin: 4px 0; }
.f { margin: 10px 0 0; } .f b { display: inline-block; min-width: 92px; }
.line { border-bottom: 1px solid #999; height: 30px; } .rule { font-size: 8.5pt; color: #7a0c2e; }
table { border-collapse: collapse; font-size: 9pt; width: 100%; } td, th { border: 1px solid #aaa; padding: 3px 5px; vertical-align: top; }
.tag { display: inline-block; background: #fbeee6; border: 1px solid #c56; padding: 1px 6px; border-radius: 3px; font-size: 9pt; }
"""

FIELDS = [
    ("A (edited)", "Edit the extracted question until it is faithful to the paper. If it is already right, copy it.", 2),
    ("B", "Same question, different words. Share at most 30 % of A's words (Jaccard &le; 0.3).", 3),
    ("B_min", "Same question, tiny edit: keep at least 75 % of A's words (e.g. change one verb).", 2),
    ("C", "Near-miss of the type shown above: almost the same words, but a DIFFERENT question.", 2),
    ("C_tilde", "C's meaning in different words: Jaccard with A &le; 0.3.", 3),
    ("narrower", "A special case of A (add a dataset, model, domain...).", 2),
    ("broader", "A question that contains A as a special case.", 2),
]


def cmd_pdf():
    import subprocess
    rows = read_jsonl(HERE / "workbook_items.jsonl")
    ex = read_jsonl(HERE / "contrast.jsonl")[0]
    h = [f"<style>{CSS}</style><h1>E1 hand-writing workbook</h1>",
         "<p>75 real arXiv papers posted in 2025 (random, seed 0). For each paper, question <b>A</b> was extracted with "
         "the single fixed prompt <code>EXTRACT_RQ</code> (common/prompts.py); <b>edit it by hand</b> until it is "
         "faithful to the paper. Write every other field <b>yourself</b>. You need at least 60 complete items. Skip a "
         "paper if its question is unclear.</p>",
         "<h2>Rules (checked by <code>python E1/eval_contrast.py --check</code>)</h2><ul>"
         "<li>Jaccard(A, B) &le; 0.3 &nbsp;&middot;&nbsp; B_min keeps &ge; 75 % of A's words &nbsp;&middot;&nbsp; Jaccard(A, C_tilde) &le; 0.3</li>"
         "<li>&ge; 10 items of each near-miss type (the type is assigned on each page; 15 each)</li>"
         "<li>&ge; 10 items where <b>broader is longer than narrower</b> (marked &#9733; on the page)</li>"
         "<li>Jaccard = shared lower-case words / all distinct words of the two questions.</li></ul>",
         "<h2>Near-miss types</h2><table>" + "".join(f"<tr><td><b>{t}</b></td><td>{d}</td></tr>" for t, d in TYPE_HELP.items()) + "</table>",
         "<h2>Worked example (from the paper)</h2><table>" + "".join(
             f"<tr><td><b>{k}</b></td><td>{escape(str(ex[k]))}</td></tr>" for k in ("A", "B", "B_min", "C", "C_tilde", "narrower", "broader")) + "</table>",
         "<h2>After writing</h2><ol><li>Type your answers into <code>E1/contrast_template.jsonl</code> (same ids), save it as "
         "<code>E1/contrast.jsonl</code> (keep the h000 example line first).</li><li>Run the checker and fix what it flags.</li>"
         "<li><b>At least 24 hours later</b>, re-read every item and fix mistakes; note the date in NOTEBOOK.md.</li></ol>"
         "<div style='break-after: page'></div>"]
    for r in rows:
        star = " &nbsp;<span class='tag'>&#9733; make <b>broader</b> LONGER than <b>narrower</b></span>" if r["make_broader_longer"] else ""
        h.append(f"<div class='item'><h2>{r['id']} &nbsp;<span class='tag'>C type: {r['C_type']}</span>{star}</h2>"
                 f"<div class='meta'>arXiv {r['source_arxiv']} &middot; {r['date']}</div><b>{escape(r['title'])}</b>"
                 f"<div class='abs'>{escape(r['abstract'])}</div>"
                 f"<div class='A'><b>Extracted A:</b> {escape(r['A_extracted'] or '(not extracted yet)')}</div>"
                 f"<div class='rule'>C type: {TYPE_HELP[r['C_type']]}</div>")
        for name, rule, lines in FIELDS:
            h.append(f"<div class='f'><b>{name}</b> <span class='rule'>{rule}</span></div>" + "<div class='line'></div>" * lines)
        h.append("</div>")
    html = HERE / "workbook.html"
    html.write_text("<!doctype html><meta charset='utf-8'>" + "".join(h), encoding="utf-8")
    pdf = (HERE / "workbook.pdf").resolve()
    for b in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        try:
            subprocess.run([b, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                            html.resolve().as_uri()], check=True, capture_output=True, timeout=120)
            break
        except Exception:
            continue
    html.unlink()
    tmpl = [ex] + [{"id": r["id"], "source_arxiv": r["source_arxiv"], "A": r["A_extracted"] or "", "B": "", "B_min": "",
                    "C": "", "C_type": r["C_type"], "C_tilde": "", "narrower": "", "broader": "",
                    "length_flipped": bool(r["make_broader_longer"]), "author": "YN", "second_reader_agrees": None}
                   for r in rows]
    write_jsonl(HERE / "contrast_template.jsonl", tmpl)
    print("wrote", pdf, "and E1/contrast_template.jsonl")


if __name__ == "__main__":
    {"select": cmd_select, "extract": cmd_extract, "pdf": cmd_pdf}[sys.argv[1]]()
