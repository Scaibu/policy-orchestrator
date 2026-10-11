from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDoraWeightDecomposedLora:
    """
    ---
    contract:
      algo_id: ALGO-NN-220
      name: NnAlgoDoraWeightDecomposedLora
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - dora
        - rslora
        - weight_decomposition
        - rank_stabilization
      inputs:
        type: object
        required:
          - input_vector
          - base_weight
          - lora_a
          - lora_b
          - magnitude_vector
        properties:
          input_vector:
            type: array
            items:
              type: number
            description: Input activation vector x of dimension k.
          base_weight:
            type: array
            items:
              type: array
              items:
                type: number
            description: Pretrained frozen base weight matrix W_0 of shape (d, k).
          lora_a:
            type: array
            items:
              type: array
              items:
                type: number
            description: Directional adapter down-projection matrix A of shape (r, k).
          lora_b:
            type: array
            items:
              type: array
              items:
                type: number
            description: Directional adapter up-projection matrix B of shape (d, r).
          magnitude_vector:
            type: array
            items:
              type: number
            description: Learnable magnitude scaling vector m of dimension d.
          lora_alpha:
            type: number
            default: 16.0
            description: LoRA alpha scaling factor.
          lora_rank:
            type: integer
            default: 4
            description: Bottleneck rank r.
          use_rslora:
            type: boolean
            default: false
            description: If True, scale by alpha / sqrt(r) (rsLoRA); if False, scale by alpha / r (standard).
      outputs:
        type: object
        required:
          - output_vector
          - decomposed_directional_norms
          - effective_scaling
        properties:
          output_vector:
            type: array
            items:
              type: number
            description: DoRA adapted layer activation h = (m * V / ||V||) * x of dimension d.
          decomposed_directional_norms:
            type: array
            items:
              type: number
            description: L2 row norms of the combined directional matrix V = W_0 + delta_W.
          effective_scaling:
            type: number
            description: Active adapter scaling constant (alpha / r or alpha / sqrt(r)).
      complexity:
        time: O(d * k + r * (d + k))
        space: O(d * k)
      preconditions:
        - len(input.input_vector) > 0
        - len(input.base_weight) > 0
        - all(len(row) == len(input.input_vector) for row in input.base_weight)
        - len(input.lora_a) == input.lora_rank
        - all(len(row) == len(input.input_vector) for row in input.lora_a)
        - len(input.lora_b) == len(input.base_weight)
        - all(len(row) == input.lora_rank for row in input.lora_b)
        - len(input.magnitude_vector) == len(input.base_weight)
        - input.lora_rank > 0
        - input.lora_alpha > 0.0
      postconditions:
        - len(output.output_vector) == len(input.base_weight)
        - len(output.decomposed_directional_norms) == len(input.base_weight)
        - all(n > 0.0 for n in output.decomposed_directional_norms)
      certificate: "If magnitude vector m matches original W_0 row norms and B is 0, output matches base W_0 * x"
      compatible_adapters:
        - ADAPTER-DORA-LINEAR-MODULE
        - ADAPTER-RSLORA-STABILIZER
      related_algos:
        - ALGO-NN-218
        - ALGO-NN-219
        - ALGO-NN-221
      references:
        - "https://arxiv.org/abs/2402.09353"
        - "https://arxiv.org/abs/2312.03732"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_vector: List[float],
        base_weight: List[List[float]],
        lora_a: List[List[float]],
        lora_b: List[List[float]],
        magnitude_vector: List[float],
        lora_alpha: float = 16.0,
        lora_rank: int = 4,
        use_rslora: bool = False,
    ) -> Dict[str, Any]:
        k = len(input_vector)
        d = len(base_weight)
        if k == 0 or d == 0:
            raise ValueError("Precondition failed: input_vector and base_weight must not be empty")
        if any(len(row) != k for row in base_weight):
            raise ValueError("Precondition failed: each base_weight row must have length k")
        if len(lora_a) != lora_rank or any(len(row) != k for row in lora_a):
            raise ValueError("Precondition failed: lora_a must have shape (lora_rank, k)")
        if len(lora_b) != d or any(len(row) != lora_rank for row in lora_b):
            raise ValueError("Precondition failed: lora_b must have shape (d, lora_rank)")
        if len(magnitude_vector) != d:
            raise ValueError("Precondition failed: magnitude_vector must have length d")
        if lora_rank <= 0 or lora_alpha <= 0.0:
            raise ValueError("Precondition failed: lora_rank and lora_alpha must be positive")

        # 1. Effective scaling
        if use_rslora:
            scaling = lora_alpha / math.sqrt(float(lora_rank))
        else:
            scaling = lora_alpha / float(lora_rank)

        # 2. Form directional matrix V = W_0 + s * (B * A) and compute row norms
        directional_norms: List[float] = []
        output_vec: List[float] = []

        # Precompute BA matrix: (d, k)
        for i in range(d):
            # Row i of V: V_{ij} = W_0[i][j] + scaling * sum_r B[i][r] * A[r][j]
            v_row: List[float] = []
            sq_sum = 0.0
            for j in range(k):
                delta_ij = sum(lora_b[i][r] * lora_a[r][j] for r in range(lora_rank))
                v_ij = base_weight[i][j] + scaling * delta_ij
                v_row.append(v_ij)
                sq_sum += v_ij * v_ij

            norm_i = math.sqrt(sq_sum) if sq_sum > 1e-12 else 1e-6
            directional_norms.append(norm_i)

            # Normalized weight row: W_{ij} = m_i * (V_{ij} / norm_i)
            # Dot product with x
            m_i = magnitude_vector[i]
            dot_val = sum((m_i * v_row[j] / norm_i) * input_vector[j] for j in range(k))
            output_vec.append(dot_val)

        return {
            "output_vector": output_vec,
            "decomposed_directional_norms": directional_norms,
            "effective_scaling": scaling,
        }
