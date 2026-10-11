from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLoraLowRankAdaptation:
    """
    ---
    contract:
      algo_id: ALGO-NN-218
      name: NnAlgoLoraLowRankAdaptation
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - lora
        - low_rank_adaptation
        - parameter_efficient
      inputs:
        type: object
        required:
          - input_vector
          - base_weight
          - lora_a
          - lora_b
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
            description: Down-projection adapter matrix A of shape (r, k).
          lora_b:
            type: array
            items:
              type: array
              items:
                type: number
            description: Up-projection adapter matrix B of shape (d, r).
          lora_alpha:
            type: number
            default: 16.0
            description: Scaling parameter alpha.
          lora_rank:
            type: integer
            default: 4
            description: Adapter bottleneck rank r.
      outputs:
        type: object
        required:
          - output_vector
          - base_output_vector
          - lora_delta_vector
          - parameter_savings_ratio
        properties:
          output_vector:
            type: array
            items:
              type: number
            description: Combined output activation h = W_0 * x + (alpha / r) * B * (A * x) of dimension d.
          base_output_vector:
            type: array
            items:
              type: number
            description: Unadapted frozen base output W_0 * x of dimension d.
          lora_delta_vector:
            type: array
            items:
              type: number
            description: Low-rank adapter perturbation delta_h of dimension d.
          parameter_savings_ratio:
            type: number
            description: Ratio of trainable parameters to base parameters r*(d+k) / (d*k).
      complexity:
        time: O(d * k + r * (d + k))
        space: O(d + r)
      preconditions:
        - len(input.input_vector) > 0
        - len(input.base_weight) > 0
        - all(len(row) == len(input.input_vector) for row in input.base_weight)
        - len(input.lora_a) == input.lora_rank
        - all(len(row) == len(input.input_vector) for row in input.lora_a)
        - len(input.lora_b) == len(input.base_weight)
        - all(len(row) == input.lora_rank for row in input.lora_b)
        - input.lora_rank > 0
        - input.lora_alpha > 0.0
      postconditions:
        - len(output.output_vector) == len(input.base_weight)
        - len(output.lora_delta_vector) == len(input.base_weight)
        - 0.0 < output.parameter_savings_ratio <= 1.0
      certificate: "If matrix B is initialized to all zeros, output_vector equals base_output_vector exactly"
      compatible_adapters:
        - ADAPTER-LORA-LINEAR-LAYER
        - ADAPTER-PEFT-INFERENCE-MERGER
      related_algos:
        - ALGO-NN-219
        - ALGO-NN-220
        - ALGO-NN-221
      references:
        - "https://arxiv.org/abs/2106.09685"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_vector: List[float],
        base_weight: List[List[float]],
        lora_a: List[List[float]],
        lora_b: List[List[float]],
        lora_alpha: float = 16.0,
        lora_rank: int = 4,
    ) -> Dict[str, Any]:
        k = len(input_vector)
        if k == 0:
            raise ValueError("Precondition failed: input_vector must not be empty")
        d = len(base_weight)
        if d == 0:
            raise ValueError("Precondition failed: base_weight must not be empty")
        if any(len(row) != k for row in base_weight):
            raise ValueError("Precondition failed: each base_weight row must have length k")
        if len(lora_a) != lora_rank or any(len(row) != k for row in lora_a):
            raise ValueError("Precondition failed: lora_a must have shape (lora_rank, k)")
        if len(lora_b) != d or any(len(row) != lora_rank for row in lora_b):
            raise ValueError("Precondition failed: lora_b must have shape (d, lora_rank)")
        if lora_rank <= 0:
            raise ValueError("Precondition failed: lora_rank must be positive")
        if lora_alpha <= 0.0:
            raise ValueError("Precondition failed: lora_alpha must be positive")

        scaling = lora_alpha / float(lora_rank)

        # 1. Base forward: W_0 * x
        base_out: List[float] = []
        for i in range(d):
            val = sum(base_weight[i][j] * input_vector[j] for j in range(k))
            base_out.append(val)

        # 2. LoRA forward: A * x of length r
        ax: List[float] = []
        for r_idx in range(lora_rank):
            val = sum(lora_a[r_idx][j] * input_vector[j] for j in range(k))
            ax.append(val)

        # 3. LoRA forward: s * B * (A * x) of length d
        lora_delta: List[float] = []
        for i in range(d):
            val = scaling * sum(lora_b[i][r_idx] * ax[r_idx] for r_idx in range(lora_rank))
            lora_delta.append(val)

        # 4. Total output: h = W_0 * x + delta
        output_vec = [base_out[i] + lora_delta[i] for i in range(d)]

        # Parameter efficiency ratio
        trainable_params = lora_rank * (d + k)
        base_params = d * k
        savings_ratio = float(trainable_params) / float(base_params)

        return {
            "output_vector": output_vec,
            "base_output_vector": base_out,
            "lora_delta_vector": lora_delta,
            "parameter_savings_ratio": savings_ratio,
        }
