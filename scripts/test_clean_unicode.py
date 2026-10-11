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
        r"\mathcal{N}": "𝒩",
        r"\mathcal{L}": "ℒ",
        r"\mathcal{O}": "𝒪",
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
    s = s.replace("ℝ^ ", "ℝ^D ")
    if s.endswith("ℝ^") or s.endswith("ℝ"):
        s = s.rstrip(" ")
        if s.endswith("ℝ^"):
            s = s + "D"

    # Remove any leftover lone backslashes or unknown latex command names
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

test_strings = [
    "Clean state x_0 ∈ ℝ^",
    "Noise \\boldsymbolε ∈",
    "Prediction \\boldsymb",
    "s_α ≥ts \\sqrtα_t, s_σ ≥t",
    "x_t ≥ts zeros(D), x_0 ≥t",
    "Reconstructed x_0 ∈ ℝ^",
    "Primary Formulation: q(x_1:T \\mid x_0) = ∏_t=1^T q(x_t \\mid x_t-1), q(x_t \\mid x_t-1) = N≤ft( x_t; \\sqrt1"
]
for t in test_strings:
    print("IN :", t)
    print("OUT:", clean_latex_to_unicode(t))
    print()
