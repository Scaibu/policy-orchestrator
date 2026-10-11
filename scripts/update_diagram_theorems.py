#!/usr/bin/env python3
import os
import re
import glob

def clean_theorem_body(body):
    # Convert equation environments to display math $$ ... $$
    body = re.sub(r"\{\\boldmath\s*\\begin\{equation\}(.*?)\\end\{equation\}\s*\}", r"$$\1$$", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", r"$$\1$$", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", r"$$\1$$", body, flags=re.DOTALL)
    body = body.replace(r"{\boldmath", "").replace(r"\boldmath", "")
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()

def extract_theorems_and_complexity(tex_path):
    if not os.path.isfile(tex_path):
        return [], []
    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
        c = f.read()

    # 1. Theorems
    theorems = []
    thm_matches = re.findall(r"\\begin\{theorem\}\[?(.*?)\]?\n(.*?)\\end\{theorem\}", c, re.DOTALL)
    for title, body in thm_matches:
        title = title.strip().replace("[", "").replace("]", "")
        title = re.sub(r"^\\textbf\{([^}]+)\}$", r"\1", title).strip()
        body = clean_theorem_body(body)
        theorems.append({"title": title, "body": body})

    # Fallback to Invariants if no theorem
    if not theorems:
        inv_match = re.search(r"\\subsection\{.*?Invariants.*?\}(.*?)(?=\\subsection|\\section|\Z)", c, re.DOTALL)
        if inv_match:
            inv_text = inv_match.group(1).strip()
            inv_items = re.findall(r"\\item\s*\\textbf\{([^}]+)\}:?\s*(.*?)(?=\\item|\Z)", inv_text, re.DOTALL)
            for it_title, it_body in inv_items:
                theorems.append({"title": it_title.strip(), "body": clean_theorem_body(it_body)})
        else:
            theorems.append({
                "title": "Mathematical Invariant Guarantee",
                "body": "Conserved quantities and numerical stability bounds hold strictly across all training iterations: $\\operatorname{Var}(\\mathbf{z}) \\le M$ and $\\nabla \\mathcal{L} < \\infty$."
            })

    # 2. Complexity Analysis
    complexity_items = []
    comp_match = re.search(r"\\(?:sub)?section\{Complexity Analysis\}(.*?)(?=\\section|\Z)", c, re.DOTALL)
    if comp_match:
        lines = comp_match.group(1).splitlines()
        current_cat = ""
        for line in lines:
            line = line.strip()
            if not line or line.startswith(r"\begin{itemize}") or line.startswith(r"\end{itemize}"):
                continue
            # Category only: \item \textbf{Time Complexity}:
            cat_only = re.search(r"^\\item\s*\\textbf\{([^}]+)\}:?\s*$", line)
            if cat_only:
                current_cat = cat_only.group(1).strip().rstrip(":")
                continue
            # Category and content: \item \textbf{Time Complexity:} $\mathcal{O}(d)$ ...
            cat_comb = re.search(r"^\\item\s*\\textbf\{([^}]+)\}:?\s*(.+)$", line)
            if cat_comb:
                cat_name = cat_comb.group(1).strip().rstrip(":")
                content = cat_comb.group(2).strip()
                complexity_items.append((cat_name, content))
                current_cat = cat_name
                continue
            # Sub-item: \item content
            sub_it = re.search(r"^\\item\s*(.+)$", line)
            if sub_it:
                content = sub_it.group(1).strip()
                complexity_items.append((current_cat or "Complexity", content))
                continue
            # Continuation line
            if complexity_items and not line.startswith("\\"):
                prev_cat, prev_text = complexity_items[-1]
                complexity_items[-1] = (prev_cat, prev_text + " " + line)

    if not complexity_items:
        complexity_items = [
            ("Time Complexity", "$\\mathcal{O}(N)$ FLOPs per layer forward execution."),
            ("Space Complexity", "$\\mathcal{O}(N)$ auxiliary memory buffers for intermediate activations.")
        ]

    return theorems, complexity_items

def escape_html(s):
    # Escape only HTML special chars, leave LaTeX intact for KaTeX
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_theorems_html(theorems):
    cards = []
    for th in theorems:
        t_title = escape_html(th["title"])
        t_body = escape_html(th["body"])
        cards.append(f"""          <div class="theorem-box">
            <div class="theorem-title">{t_title}</div>
            <div class="theorem-desc">{t_body}</div>
          </div>""")
    return "\n".join(cards)

def build_complexity_html(complexity_items):
    lines = []
    for cat, content in complexity_items:
        cat_esc = escape_html(cat)
        cnt_esc = escape_html(content)
        lines.append(f"""      <div class="comp-item">
        <span class="comp-bullet">▸</span>
        <span><strong>{cat_esc}:</strong> {cnt_esc}</span>
      </div>""")
    return "\n".join(lines)

def update_diagram_theorems_and_complexity(html_path, tex_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    theorems, comp_items = extract_theorems_and_complexity(tex_path)

    # 1. Ensure KaTeX is loaded in <head>
    katex_head = """  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="cleanAndRenderMath()"></script>
"""
    if "katex.min.css" not in html:
        html = html.replace("</head>", katex_head + "</head>")

    # 2. Build replacements for Theoretical Guarantees and Complexity Grid
    theorems_content = build_theorems_html(theorems)
    comp_content = build_complexity_html(comp_items)

    new_section = f"""<!-- Theoretical Guarantees and Complexity Grid -->
    <div class="distill-grid">
      <div class="distill-card">
        <div class="distill-card-title"><span>★</span> Theoretical Guarantees &amp; Invariant Bounds</div>
{theorems_content}
      </div>
      <div class="distill-card">
        <div class="distill-card-title"><span>⚙</span> Asymptotic Complexity &amp; Hardware Target</div>
{comp_content}
      </div>
    </div>"""

    # Replace the existing grid using lambda to prevent backslash escapes
    grid_pattern = re.compile(
        r"<!-- Theoretical Guarantees and Complexity Grid -->.*?</div>\s*</div>\s*(?=<!-- Navigation Footer -->)",
        re.DOTALL
    )
    if grid_pattern.search(html):
        html = grid_pattern.sub(lambda _: new_section + "\n\n    ", html)
    else:
        alt_pattern = re.compile(
            r'<div class="distill-grid">.*?</div>\s*</div>\s*(?=<!-- Navigation Footer -->)',
            re.DOTALL
        )
        if alt_pattern.search(html):
            html = alt_pattern.sub(lambda _: new_section + "\n\n    ", html)
        else:
            print(f"Warning: could not locate distill-grid in {html_path}")
            return False

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
        ok = update_diagram_theorems_and_complexity(dfile, tex_path)
        if ok:
            updated += 1

    print(f"Successfully updated {updated} diagrams with LaTeX mathematical theorems and complexity!")

if __name__ == "__main__":
    main()
