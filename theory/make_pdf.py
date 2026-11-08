"""Render theory/*.md (with $...$ / $$...$$ maths) to PDF.

    python theory/make_pdf.py

Markdown -> HTML (python-markdown), maths typeset by KaTeX, printed to PDF by headless Edge/Chrome.
"""
import re
import subprocess
import sys
from html import escape
from pathlib import Path

import markdown

HERE = Path(__file__).parent
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "chromium", "google-chrome", "msedge"]

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
@page {{ size: A4; margin: 18mm 17mm; }}
body {{ font-family: "Georgia", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45; color: #111; }}
h1 {{ font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
h2 {{ font-size: 14pt; margin-top: 22px; break-before: page; color: #7a0c2e; }}
h2:first-of-type {{ break-before: avoid; }}
h3 {{ font-size: 11.5pt; margin: 14px 0 4px; }}
table {{ border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; width: 100%; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; vertical-align: top; text-align: left; }}
th {{ background: #eee; }}
code {{ font-family: Consolas, monospace; font-size: 9pt; }}
hr {{ display: none; }}
li {{ margin: 2px 0; }}
.katex-display {{ margin: 8px 0; }}
.sheet h2 {{ break-before: auto; border-bottom: 1px solid #7a0c2e; }}
.sheet li {{ margin: 6px 0; }}
</style></head><body class="{cls}">{body}
<script>
document.addEventListener("DOMContentLoaded", function () {{
  document.querySelectorAll(".m").forEach(function (el) {{
    katex.render(el.textContent, el, {{displayMode: el.dataset.d === "1", throwOnError: false}});
  }});
  document.body.setAttribute("data-done", "1");
}});
</script></body></html>"""


def md_to_html(md):
    math = []

    def keep(m, display):
        math.append((m.group(1), display))
        return f"@@MATH{len(math) - 1}@@"

    md = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(m, True), md, flags=re.S)
    md = re.sub(r"(?<![\\$])\$(?!\s)(.+?)(?<!\s)\$", lambda m: keep(m, False), md)
    html = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html = re.sub(r"\[([^\]]+)\]\(([^)]+\.md)\)", r"\1", html)

    def put(m):
        tex, d = math[int(m.group(1))]
        tag = "div" if d else "span"
        return f'<{tag} class="m" data-d="{int(d)}">{escape(tex)}</{tag}>'

    return re.sub(r"@@MATH(\d+)@@", put, html)


def render(md_path, pdf_path, title, cls=""):
    md = md_path.read_text(encoding="utf-8")
    md = md.replace("[answer_sheets.md](answer_sheets.md)", "the answer sheets (answer_sheets.pdf)")
    md = md.replace("[answers.md](answers.md)", "the full working (working.pdf)")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(TEMPLATE.format(title=title, body=md_to_html(md), cls=cls), encoding="utf-8")
    for b in BROWSERS:
        try:
            subprocess.run([b, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path.resolve()}",
                            html_path.resolve().as_uri()], check=True, capture_output=True, timeout=120)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    else:
        sys.exit("No Chromium-based browser found")
    html_path.unlink()
    print("wrote", pdf_path)


if __name__ == "__main__":
    render(HERE / "answers.md", HERE / "working.pdf", "Theory paper - full working (T1-T5)")
    render(HERE / "answer_sheets.md", HERE / "answer_sheets.pdf", "Theory paper - answer sheets", cls="sheet")
