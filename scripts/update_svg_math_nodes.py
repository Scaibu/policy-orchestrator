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

def parse_require_ensure_safe(raw_str):
    if not raw_str:
        return []
    items = []
    current = []
    in_math = False
    brace_depth = 0
    bracket_depth = 0
    for ch in raw_str:
        if ch == "$":
            in_math = not in_math
            current.append(ch)
        elif ch == "{" and not in_math:
            brace_depth += 1
            current.append(ch)
        elif ch == "}" and not in_math:
            brace_depth = max(0, brace_depth - 1)
            current.append(ch)
        elif ch == "[" and not in_math:
            bracket_depth += 1
            current.append(ch)
        elif ch == "]" and not in_math:
            bracket_depth = max(0, bracket_depth - 1)
            current.append(ch)
        elif ch == "," and not in_math and brace_depth == 0 and bracket_depth == 0:
            item_str = "".join(current).strip()
            if item_str:
                items.append(item_str)
            current = []
        else:
            current.append(ch)
    if current:
        item_str = "".join(current).strip()
        if item_str:
            items.append(item_str)
    return items

def to_unicode_math(latex_str, max_len=26):
    if not latex_str:
        return ""
    s = latex_str.replace("$", "")
    syms = [
        (r"\\mathbf\{([a-zA-Z0-9_]+)\}", r"\1"),
        (r"\\boldsymbol\{([a-zA-Z0-9_]+)\}", r"\1"),
        (r"\\mathrm\{([^}]+)\}", r"\1"),
        (r"\\text\{([^}]+)\}", r"\1"),
        (r"\\mathbb\{R\}\^\{?([^}]+)\}?", r"ℝ^\1"),
        (r"\\mathbb\{R\}", "ℝ"),
        (r"\\in", "∈"),
        (r"\\gets", "←"),
        (r"\\leftarrow", "←"),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\mathcal\{F\}", "ℱ"),
        (r"\\mathcal\{L\}", "ℒ"),
        (r"\\mathcal\{N\}", "𝒩"),
        (r"\\Delta", "Δ"),
        (r"\\sigma", "σ"),
        (r"\\mu", "μ"),
        (r"\\epsilon", "ε"),
        (r"\\theta", "θ"),
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\sqrt\{([^}]+)\}", r"√(\1)"),
        (r"\\sqrt([a-zA-Z0-9α-ωΑ-Ω])", r"√\1"),
        (r"\^\{([^}]+)\}", r"^\1"),
        (r"_\{([^}]+)\}", r"_\1"),
        (r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r"∑"),
        (r"\\sum", "∑"),
        (r"\\prod_\{([^}]+)\}\^\{([^}]+)\}", r"∏"),
        (r"\\prod", "∏"),
    ]
    for p, r in syms:
        s = re.sub(p, r, s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(r"\boldsymbol", "").replace(r"\boldsymb", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len-1].rstrip() + "…"
    return s

def format_node_2lines(raw_item, default_sub="Tensor"):
    raw = raw_item.strip()
    m = re.search(r"(\$[^$]+\$)", raw)
    if m:
        math_part = m.group(1)
        name_part = raw[:m.start()].strip().rstrip(":")
        clean_math = to_unicode_math(math_part, max_len=26)
        clean_name = to_unicode_math(name_part, max_len=20)
        return clean_math or clean_name, clean_name or default_sub
    else:
        clean = to_unicode_math(raw, max_len=26)
        return clean, default_sub

def escape_xml(s):
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))

def parse_tex_elements(tex_path):
    if not os.path.isfile(tex_path):
        return None
    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
        c = f.read()

    # 1. Require (Inputs)
    req_m = re.search(r"\\Require\s+([^\n\r]+)", c)
    req_items = parse_require_ensure_safe(req_m.group(1)) if req_m else []

    # 2. Ensure (Outputs)
    ens_m = re.search(r"\\Ensure\s+([^\n\r]+)", c)
    ens_items = parse_require_ensure_safe(ens_m.group(1)) if ens_m else []

    # 3. States (Kernel & Activation operations)
    st_matches = re.findall(r"\\State\s+([^\n\r]+)", c)
    states = [s.strip().replace("$", "") for s in st_matches if not s.strip().startswith(r"\Return")]

    # 4. Primary Equation
    eq_m = re.search(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", c, re.DOTALL)
    if not eq_m:
        eq_m = re.search(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", c, re.DOTALL)
    primary_eq = clean_latex(eq_m.group(1)) if eq_m else ""

    return {
        "req": req_items,
        "ens": ens_items,
        "states": states,
        "primary_eq": primary_eq
    }

def update_diagram_svg_math(html_path, tex_info):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    svg_m = re.search(r'(<svg[^>]*class="distill-svg-art"[^>]*>.*?</svg>)', html, re.DOTALL)
    if not svg_m:
        return False

    old_svg = svg_m.group(1)
    new_svg = old_svg

    # Prepare Input 1, 2, 3
    in1_math, in1_sub = "x_0 ∈ ℝ^D", "Batch Tensor"
    in2_math, in2_sub = "W ∈ ℝ^{D_h × D_v}", "Weights Matrix"
    in3_math, in3_sub = "b ∈ ℝ^{D_h}", "Bias / Param"

    if tex_info and tex_info["req"]:
        if len(tex_info["req"]) > 0:
            in1_math, in1_sub = format_node_2lines(tex_info["req"][0], "Batch Tensor")
        if len(tex_info["req"]) > 1:
            in2_math, in2_sub = format_node_2lines(tex_info["req"][1], "Weights Matrix")
        if len(tex_info["req"]) > 2:
            in3_math, in3_sub = format_node_2lines(tex_info["req"][2], "Bias / Param")

    # Prepare Operator Kernel (Stage 2) and Activation (Stage 3)
    k_line1 = "∑"
    k_line2 = "Linear Projection"
    k_line3 = "z = Wᵀx + b"
    act_line1 = "g(·)"
    act_line2 = "Nonlinear Gate"
    act_line3 = "a = σ(z)"

    if tex_info and tex_info["states"]:
        if len(tex_info["states"]) > 0:
            k_line2 = to_unicode_math(tex_info["states"][0], max_len=22)
        if len(tex_info["states"]) > 1:
            act_line2 = to_unicode_math(tex_info["states"][1], max_len=22)
        if len(tex_info["states"]) > 2:
            act_line3 = to_unicode_math(tex_info["states"][2], max_len=18)

    # Prepare Outputs (Stage 4)
    out1_math, out1_sub = "ŷ ∈ ℝ^D", "Target State"
    out2_math, out2_sub = "ℒ", "Invariant Bound"

    if tex_info and tex_info["ens"]:
        if len(tex_info["ens"]) > 0:
            out1_math, out1_sub = format_node_2lines(tex_info["ens"][0], "Target State")
        if len(tex_info["ens"]) > 1:
            out2_math, out2_sub = format_node_2lines(tex_info["ens"][1], "Invariant Bound")

    # Prepare Footer Primary Equation
    footer_eq = ""
    if tex_info and tex_info["primary_eq"]:
        footer_eq = to_unicode_math(tex_info["primary_eq"], max_len=75)
    if not footer_eq:
        footer_eq = "State Transition & Invariant Guarantee Formulation"

    # Replace Input 1 (x=90, y=105 is math, y=125 is sub)
    new_svg = re.sub(
        r'(<text x="90" y="105" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in1_math) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="90" y="125" fill="#94a3b8" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in1_sub) + r'\g<2>',
        new_svg
    )

    # Replace Input 2 (x=90, y=190 is math, y=210 is sub)
    new_svg = re.sub(
        r'(<text x="90" y="190" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in2_math) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="90" y="210" fill="#94a3b8" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in2_sub) + r'\g<2>',
        new_svg
    )

    # Replace Input 3 (x=90, y=275 is math, y=295 is sub)
    new_svg = re.sub(
        r'(<text x="90" y="275" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in3_math) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="90" y="295" fill="#f59e0b" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(in3_sub) + r'\g<2>',
        new_svg
    )

    # Replace Stage 2 (Operator Kernel)
    new_svg = re.sub(
        r'(<text x="410" y="220" fill="#ffffff" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(k_line2) + r'\g<2>',
        new_svg
    )

    # Replace Stage 3 (Activation)
    new_svg = re.sub(
        r'(<text x="670" y="220" fill="#ffffff" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(act_line2) + r'\g<2>',
        new_svg
    )
    if act_line3:
        new_svg = re.sub(
            r'(<text x="670" y="251" fill="#93c5fd" [^>]*>)[^<]*(</text>)',
            r'\g<1>' + escape_xml(act_line3) + r'\g<2>',
            new_svg
        )

    # Replace Stage 4 Output 1
    new_svg = re.sub(
        r'(<text x="890" y="150" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out1_math) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="890" y="170" fill="#10b981" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out1_sub) + r'\g<2>',
        new_svg
    )

    # Replace Stage 4 Output 2
    new_svg = re.sub(
        r'(<text x="890" y="235" fill="#f8fafc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out2_math) + r'\g<2>',
        new_svg
    )
    new_svg = re.sub(
        r'(<text x="890" y="255" fill="#c084fc" [^>]*>)[^<]*(</text>)',
        r'\g<1>' + escape_xml(out2_sub) + r'\g<2>',
        new_svg
    )

    # Replace Footer Formula Banner inside SVG
    new_svg = re.sub(
        r'(<text x="580" y="374" fill="#38bdf8" [^>]*>)[^<]*(</text>)',
        r'\g<1>Primary Formulation: ' + escape_xml(footer_eq) + r'\g<2>',
        new_svg
    )

    # Put updated SVG back into html
    updated_html = html[:svg_m.start(1)] + new_svg + html[svg_m.end(1):]
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(updated_html)

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
        tex_info = parse_tex_elements(tex_path)
        ok = update_diagram_svg_math(dfile, tex_info)
        if ok:
            updated += 1

    print(f"Successfully updated computational diagram SVG math in {updated} diagrams!")

if __name__ == "__main__":
    main()
