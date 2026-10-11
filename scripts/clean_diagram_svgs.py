#!/usr/bin/env python3
import os
import re
import glob

def clean_latex_to_unicode(s):
    if not s:
        return ""
    # Decode XML / HTML entities if present
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
    s = s.replace("$", "")
    
    sym_map = {
        r"\boldsymbol{\epsilon}": "ε",
        r"\boldsymbol\epsilon": "ε",
        r"\epsilon": "ε",
        r"\varepsilon": "ε",
        r"\boldsymbol{\mu}": "μ",
        r"\boldsymbol\mu": "μ",
        r"\mu": "μ",
        r"\boldsymbol{\sigma}": "σ",
        r"\sigma": "σ",
        r"\boldsymbol{\theta}": "θ",
        r"\theta": "θ",
        r"\boldsymbol{\alpha}": "α",
        r"\alpha": "α",
        r"\boldsymbol{\beta}": "β",
        r"\beta": "β",
        r"\boldsymbol{\lambda}": "λ",
        r"\lambda": "λ",
        r"\boldsymbol{\gamma}": "γ",
        r"\gamma": "γ",
        r"\boldsymbol{\delta}": "δ",
        r"\delta": "δ",
        r"\boldsymbol{\Delta}": "Δ",
        r"\Delta": "Δ",
        r"\boldsymbol{\Omega}": "Ω",
        r"\Omega": "Ω",
        r"\gets": "←",
        r"\leftarrow": "←",
        r"\to": "→",
        r"\rightarrow": "→",
        r"\in": "∈",
        r"\notin": "∉",
        r"\subset": "⊂",
        r"\subseteq": "⊆",
        r"\cdot": "·",
        r"\times": "×",
        r"\otimes": "⊗",
        r"\oplus": "⊕",
        r"\circ": "∘",
        r"\le": "≤",
        r"\leq": "≤",
        r"\ge": "≥",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\sim": "~",
        r"\infty": "∞",
        r"\nabla": "∇",
        r"\partial": "∂",
        r"\mid": "|",
        r"\parallel": "||",
        r"\forall": "∀",
        r"\exists": "∃",
        r"\top": "⊤",
        r"\bot": "⊥",
        r"\dagger": "†",
        r"\mathbb{R}": "ℝ",
        r"\mathbb{C}": "ℂ",
        r"\mathbb{Z}": "ℤ",
        r"\mathbb{N}": "ℕ",
        r"\mathcal{N}": "N",
        r"\mathcal{L}": "L",
        r"\mathcal{O}": "O",
        r"\pm": "±",
        r"\mp": "∓",
        r"\dots": "...",
        r"\ldots": "...",
        r"\cdots": "...",
        r"\quad": " ",
        r"\qquad": "  ",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
    }
    for k, v in sym_map.items():
        s = s.replace(k, v)

    # Regex patterns
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathit\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\hat\{([^}]+)\}", r"\1̂", s)
    s = re.sub(r"\\bar\{([^}]+)\}", r"\1̄", s)
    s = re.sub(r"\\tilde\{([^}]+)\}", r"\1̃", s)
    s = re.sub(r"\\vec\{([^}]+)\}", r"\1⃗", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt([a-zA-Z0-9α-ωΑ-Ω])", r"√\1", s)
    s = re.sub(r"\\mathbb\{R\}\^\{?([^}]+)\}?", r"ℝ^\1", s)
    s = re.sub(r"\\left[\(\[\{]", "(", s)
    s = re.sub(r"\\right[\)\]\}]", ")", s)
    s = re.sub(r"\\prod_\{([^}]+)\}\^\{([^}]+)\}", r"∏_{\1}^{\2}", s)
    s = re.sub(r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r"∑_{\1}^{\2}", s)
    s = re.sub(r"\\prod", "∏", s)
    s = re.sub(r"\\sum", "∑", s)

    # Artifact cleanup
    s = s.replace(r"\boldsymbol", "")
    s = s.replace(r"\boldsymb", "")
    s = s.replace(r"\bold", "")
    s = s.replace("≥ts", "←")
    s = s.replace("≤ft", "")
    
    # Clean ℝ^ without dimension
    s = re.sub(r"ℝ\^\s+", "ℝ^D ", s)
    if s.endswith("ℝ^") or s.endswith("ℝ"):
        s = s.rstrip(" ")
        if s.endswith("ℝ^"):
            s = s + "D"

    # Remove any leftover lone backslashes or unknown latex command names
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def escape_xml(s):
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))

def parse_tex_file(tex_path):
    if not os.path.isfile(tex_path):
        return None
    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
        c = f.read()
    
    req_m = re.search(r"\\Require\s+([^\n\r]+)", c)
    ens_m = re.search(r"\\Ensure\s+([^\n\r]+)", c)
    states = re.findall(r"\\State\s+([^\n\r]+)", c)
    eq_m = re.search(r"\\begin\{equation\}(.*?)\\end\{equation\}", c, re.DOTALL)
    
    # Process inputs (Require)
    inputs = []
    if req_m:
        raw_req = req_m.group(1).strip()
        # split by comma where comma is not inside brackets/parentheses
        parts = re.split(r",\s*(?![^{}]*\})", raw_req)
        for p in parts:
            p_clean = clean_latex_to_unicode(p)
            if p_clean:
                inputs.append(p_clean)
                
    # Process outputs (Ensure)
    outputs = []
    if ens_m:
        raw_ens = ens_m.group(1).strip()
        parts = re.split(r",\s*(?![^{}]*\})", raw_ens)
        for p in parts:
            p_clean = clean_latex_to_unicode(p)
            if p_clean:
                outputs.append(p_clean)

    # Process operations (States)
    ops = []
    for st in states:
        clean_st = clean_latex_to_unicode(st)
        if clean_st and not clean_st.startswith("return") and not clean_st.startswith("Return"):
            ops.append(clean_st)

    # Primary equation
    primary_eq = ""
    if eq_m:
        primary_eq = clean_latex_to_unicode(eq_m.group(1))
        
    return {
        "inputs": inputs,
        "outputs": outputs,
        "ops": ops,
        "primary_eq": primary_eq
    }

def update_diagram_svg(diagram_path, tex_info):
    with open(diagram_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Find the SVG inside diagram-panel-box
    svg_m = re.search(r'(<svg[^>]*class="distill-svg-art"[^>]*>.*?</svg>)', html, re.DOTALL)
    if not svg_m:
        return False, "No SVG found"

    old_svg = svg_m.group(1)

    # Prepare labels
    # Inputs: up to 3 inputs
    in1_txt = "Input State x₀ ∈ ℝ^D"
    in2_txt = "Weights Matrix W"
    in3_txt = "Bias / Parameters b"
    if tex_info and tex_info["inputs"]:
        if len(tex_info["inputs"]) > 0:
            in1_txt = tex_info["inputs"][0][:25].strip()
        if len(tex_info["inputs"]) > 1:
            in2_txt = tex_info["inputs"][1][:25].strip()
        if len(tex_info["inputs"]) > 2:
            in3_txt = tex_info["inputs"][2][:25].strip()

    # Kernel & Act operations
    op_kernel = "z = Wᵀx + b"
    op_kernel_math = "∑ Linear Projection"
    op_act = "Nonlinear Gate"
    op_act_math = "g(·) Activation"
    if tex_info and tex_info["ops"]:
        if len(tex_info["ops"]) > 0:
            op_kernel_math = tex_info["ops"][0][:22].strip()
        if len(tex_info["ops"]) > 1:
            op_act_math = tex_info["ops"][1][:22].strip()

    # Outputs: up to 2 outputs
    out1_txt = "Output State ŷ ∈ ℝ^D"
    out2_txt = "Loss / Invariant Bound ℒ"
    if tex_info and tex_info["outputs"]:
        if len(tex_info["outputs"]) > 0:
            out1_txt = tex_info["outputs"][0][:28].strip()
        if len(tex_info["outputs"]) > 1:
            out2_txt = tex_info["outputs"][1][:28].strip()

    # Primary formula
    formula_txt = ""
    if tex_info and tex_info["primary_eq"]:
        formula_txt = tex_info["primary_eq"]
        if len(formula_txt) > 85:
            formula_txt = formula_txt[:82] + "…"
    if not formula_txt:
        formula_txt = "State Transition & Invariant Guarantee Formulation"

    # Now let's perform targeted replacements in old_svg
    new_svg = old_svg

    # 1. Expand Stage 1 rect width from 160 to 200
    new_svg = re.sub(
        r'<rect x="50" y="80" width="160" height="64"',
        r'<rect x="40" y="80" width="200" height="64"',
        new_svg
    )
    new_svg = re.sub(
        r'<rect x="50" y="165" width="160" height="64"',
        r'<rect x="40" y="165" width="200" height="64"',
        new_svg
    )
    new_svg = re.sub(
        r'<rect x="50" y="250" width="160" height="64"',
        r'<rect x="40" y="250" width="200" height="64"',
        new_svg
    )

    # Shift circle cx from 75 to 65
    new_svg = re.sub(r'<circle cx="75" cy="112" r="14"', r'<circle cx="65" cy="112" r="14"', new_svg)
    new_svg = re.sub(r'<circle cx="75" cy="197" r="14"', r'<circle cx="65" cy="197" r="14"', new_svg)
    new_svg = re.sub(r'<circle cx="75" cy="282" r="14"', r'<circle cx="65" cy="282" r="14"', new_svg)
    new_svg = re.sub(r'<text x="75" y="117"', r'<text x="65" y="117"', new_svg)
    new_svg = re.sub(r'<text x="75" y="202"', r'<text x="65" y="202"', new_svg)
    new_svg = re.sub(r'<text x="75" y="287"', r'<text x="65" y="287"', new_svg)

    # Shift input text x from 100 to 90
    new_svg = re.sub(r'<text x="100" y="105"', r'<text x="90" y="105"', new_svg)
    new_svg = re.sub(r'<text x="100" y="125"', r'<text x="90" y="125"', new_svg)
    new_svg = re.sub(r'<text x="100" y="190"', r'<text x="90" y="190"', new_svg)
    new_svg = re.sub(r'<text x="100" y="210"', r'<text x="90" y="210"', new_svg)
    new_svg = re.sub(r'<text x="100" y="275"', r'<text x="90" y="275"', new_svg)
    new_svg = re.sub(r'<text x="100" y="295"', r'<text x="90" y="295"', new_svg)

    # Adjust path starting x from 210 to 240
    new_svg = re.sub(r'M 210 112 C 260 112, 270 160, 310 180', r'M 240 112 C 275 112, 285 160, 310 180', new_svg)
    new_svg = re.sub(r'M 210 197 C 260 197, 270 200, 310 200', r'M 240 197 C 275 197, 285 200, 310 200', new_svg)
    new_svg = re.sub(r'M 210 282 C 260 282, 270 230, 310 215', r'M 240 282 C 275 282, 285 230, 310 215', new_svg)

    # 2. Update Input 1, 2, 3 text content cleanly
    new_svg = re.sub(
        r'(<text x="90" y="105" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in1_txt) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="90" y="190" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in2_txt) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="90" y="275" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in3_txt) + r'\g<2>',
        new_svg
    )

    # 3. Update Stage 2 text (Operator Kernel)
    new_svg = re.sub(
        r'(<text x="410" y="220" fill="#ffffff" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(op_kernel_math) + r'\g<2>',
        new_svg
    )

    # 4. Update Stage 3 text (Activation / Gate)
    new_svg = re.sub(
        r'(<text x="670" y="220" fill="#ffffff" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(op_act_math) + r'\g<2>',
        new_svg
    )

    # 5. Update Stage 4 text (Outputs)
    new_svg = re.sub(
        r'(<text x="890" y="150" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out1_txt) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="890" y="235" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out2_txt) + r'\g<2>',
        new_svg
    )

    # 6. Update Footer Formula Banner
    # Expand banner width and center text
    new_svg = re.sub(
        r'<rect x="180" y="352" width="660" height="34" rx="6"',
        r'<rect x="140" y="352" width="880" height="34" rx="6"',
        new_svg
    )
    new_svg = re.sub(
        r'<text x="510" y="374" fill="#38bdf8" font-size="13"',
        r'<text x="580" y="374" fill="#38bdf8" font-size="12"',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="580" y="374" fill="#38bdf8" [^>]*>)[^<]*(</text>)',
        r'\g<1>Primary Formulation: ' + escape_xml(formula_txt) + r'\g<2>',
        new_svg
    )

    # Perform a global cleanup on any remaining LaTeX artifacts in the entire SVG
    def clean_text_match(m):
        raw = m.group(1)
        cleaned = clean_latex_to_unicode(raw)
        return f'>{escape_xml(cleaned)}<'

    # Clean text between tags
    new_svg = re.sub(r'>([^<]+)<', clean_text_match, new_svg)

    # Replace SVG in html
    updated_html = html[:svg_m.start(1)] + new_svg + html[svg_m.end(1):]

    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(updated_html)

    return True, "Updated successfully"

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_nn = os.path.join(root_dir, "src/features/code_engine/algos/nn")
    diagrams_dir = os.path.join(base_nn, "visualizer/diagrams")
    
    diagram_files = sorted(glob.glob(os.path.join(diagrams_dir, "*.html")))
    print(f"Total diagram html files: {len(diagram_files)}")
    
    updated_count = 0
    for dfile in diagram_files:
        basename = os.path.basename(dfile)
        if basename == "index.html":
            continue
        # Format: domain__algo.html
        name_part = basename[:-5]
        if "__" not in name_part:
            continue
        domain, algo = name_part.split("__", 1)
        tex_path = os.path.join(base_nn, domain, algo, "math.tex")
        
        tex_info = parse_tex_file(tex_path)
        ok, msg = update_diagram_svg(dfile, tex_info)
        if ok:
            updated_count += 1

    print(f"Successfully processed and updated {updated_count} diagrams.")

if __name__ == "__main__":
    main()
