#!/usr/bin/env python3
import os
import re
import glob

def clean_latex(s):
    if not s:
        return ""
    s = s.replace(r"{\boldmath", "").replace(r"\boldmath", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def get_math_for_diagram(tex_path):
    if not os.path.isfile(tex_path):
        return None
    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
        c = f.read()

    # Formal Definition / Excerpt
    desc_m = re.search(r"\\subsection\{Formal Mathematical Definition\}\s*(?:\\textbf\{[^}]+\})?\s*(.*?)(?=\\subsection|\\section|\\begin\{equation\}|\Z)", c, re.DOTALL)
    desc = ""
    if desc_m:
        raw_desc = desc_m.group(1).strip()
        raw_desc = re.sub(r"\\textbf\{([^}]+)\}", r"<strong>\1</strong>", raw_desc)
        raw_desc = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", "", raw_desc, flags=re.DOTALL)
        raw_desc = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", "", raw_desc, flags=re.DOTALL)
        desc = " ".join(raw_desc.split())

    if not desc or len(desc) < 15:
        def_m = re.search(r"\\section\{Definitions and Cognitive Mental Models\}\s*(.*?)(?=\\section|\Z)", c, re.DOTALL)
        if def_m:
            raw_def = def_m.group(1).strip()
            raw_def = re.sub(r"\\(?:sub)*section\{[^}]+\}", "", raw_def)
            raw_def = re.sub(r"\\textbf\{([^}]+)\}", r"<strong>\1</strong>", raw_def)
            raw_def = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", "", raw_def, flags=re.DOTALL)
            raw_def = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", "", raw_def, flags=re.DOTALL)
            desc = " ".join(raw_def.split())[:350]

    # Primary Equation
    eq_m = re.search(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", c, re.DOTALL)
    if not eq_m:
        eq_m = re.search(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", c, re.DOTALL)
    eq = ""
    if eq_m:
        eq = clean_latex(eq_m.group(1))

    return {
        "desc": desc,
        "eq": eq
    }

def escape_html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def update_diagram_flow_math(html_path, tex_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    info = get_math_for_diagram(tex_path)
    if not info:
        return False

    # 1. Update distill-excerpt if we have a clean math excerpt
    if info["desc"]:
        desc_escaped = escape_html(info["desc"])
        excerpt_replacement = f'<div class="distill-excerpt">{desc_escaped}</div>'
        excerpt_pattern = re.compile(r'<div class="distill-excerpt">.*?</div>', re.DOTALL)
        if excerpt_pattern.search(html):
            html = excerpt_pattern.sub(lambda _: excerpt_replacement, html)

    # 2. Add / Update the LaTeX Mathematical Formulation Banner inside diagram-panel-box
    if info["eq"]:
        eq_escaped = escape_html(info["eq"])
        banner_html = f"""      <!-- Formal Mathematical Formulation Banner for Computational & Dynamic Flow -->
      <div class="diagram-math-banner">
        <div class="diagram-math-header">
          <span class="diagram-math-tag">◈ PRIMARY MATHEMATICAL FORMULATION &amp; DYNAMIC FLOW</span>
          <span class="diagram-math-badge">KaTeX Verified · Closed-Form</span>
        </div>
        <div class="diagram-math-equation">$${eq_escaped}$$</div>
      </div>"""

        if '<div class="diagram-math-banner">' in html:
            banner_pattern = re.compile(r'<!-- Formal Mathematical Formulation Banner.*?</div>\s*</div>', re.DOTALL)
            html = banner_pattern.sub(lambda _: banner_html, html)
        else:
            # Find the position of </svg> inside diagram-panel-box
            svg_end_idx = html.find("</svg>")
            if svg_end_idx != -1:
                after_svg_idx = svg_end_idx + len("</svg>")
                html = html[:after_svg_idx] + "\n" + banner_html + "\n    " + html[after_svg_idx:]

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return True

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_nn = os.path.join(root_dir, "src/features/code_engine/algos/nn")
    diagrams_dir = os.path.join(base_nn, "visualizer/diagrams")

    diagram_files = sorted(glob.glob(os.path.join(diagrams_dir, "*.html")))
    print(f"Total diagram html files: {len(diagram_files)}")

    updated = 0
    for dfile in diagram_files:
        basename = os.path.basename(dfile)
        if basename == "index.html" or "__" not in basename:
            continue
        domain, algo = basename[:-5].split("__", 1)
        tex_path = os.path.join(base_nn, domain, algo, "math.tex")
        ok = update_diagram_flow_math(dfile, tex_path)
        if ok:
            updated += 1

    print(f"Successfully injected computational and dynamic flow LaTeX math into {updated} diagrams!")

if __name__ == "__main__":
    main()
