from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDifferentiableArchitectureSearchDarts:
    """
    ---
    contract:
      algo_id: ALGO-NN-186
      name: NnAlgoDifferentiableArchitectureSearchDarts
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - nas
        - darts
        - continuous_relaxation
        - architecture_search
        - bilevel_optimization
      inputs:
        type: object
        required:
          - arch_alpha_logits
          - candidate_op_outputs
        properties:
          arch_alpha_logits:
            type: array
            items:
              type: number
            description: Learnable architecture mixing parameters alpha_o of length K_ops.
          candidate_op_outputs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Candidate operation output tensors o(x) of shape (K_ops, D).
          temperature:
            type: number
            default: 1.0
            minimum: 0.0001
            description: Softmax relaxation temperature.
      outputs:
        type: object
        required:
          - mixed_edge_output
          - operation_probabilities
          - winning_op_index
          - architecture_entropy
        properties:
          mixed_edge_output:
            type: array
            items:
              type: number
            description: Continuously relaxed edge feature sum_o p_o * o(x) of dimension D.
          operation_probabilities:
            type: array
            items:
              type: number
            description: Softmax normalized architecture weights p_o of length K_ops.
          winning_op_index:
            type: integer
            description: Discrete argmax operation selection index.
          architecture_entropy:
            type: number
            description: Shannon entropy of the architecture distribution H(p) measuring search certainty.
      parameters: {}
      input_assumptions:
        - arch_alpha_logits has length K_ops >= 2
        - candidate_op_outputs has shape (K_ops, D) with D >= 1
        - temperature > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized softmax with max-subtraction"
      uses_model: false
      complexity:
        variables:
          K: number of candidate operations
          D: feature dimension
        time_worst: O(K * D)
        time_typical: O(K * D)
        space: O(D + K)
      preconditions:
        - len(input.arch_alpha_logits) >= 2
        - len(input.candidate_op_outputs) == len(input.arch_alpha_logits)
        - len(input.candidate_op_outputs[0]) > 0
        - input.temperature > 0.0
      postconditions:
        - len(output.mixed_edge_output) == len(input.candidate_op_outputs[0])
        - len(output.operation_probabilities) == len(input.arch_alpha_logits)
        - 0 <= output.winning_op_index < len(input.arch_alpha_logits)
        - output.architecture_entropy >= 0.0
      certificate: "Mixed edge output strictly equals convex combination of candidate operation activations"
      compatible_adapters:
        - ADAPTER-DARTS-CELL-ROUTER
        - ADAPTER-HARDWARE-AWARE-NAS
      related_algos:
        - ALGO-NN-184
      references:
        - "https://arxiv.org/abs/1806.09055"
    ---
    """

    @classmethod
    def forward(
        cls,
        arch_alpha_logits: List[float],
        candidate_op_outputs: List[List[float]],
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        K = len(arch_alpha_logits)
        if K < 2:
            raise ValueError("Precondition failed: at least 2 candidate operations required")
        if len(candidate_op_outputs) != K or len(candidate_op_outputs[0]) == 0:
            raise ValueError("Precondition failed: candidate_op_outputs shape mismatch")
        D = len(candidate_op_outputs[0])
        if any(len(r) != D for r in candidate_op_outputs):
            raise ValueError("Precondition failed: non-uniform operation output dimensions")
        if temperature <= 0.0:
            raise ValueError("Precondition failed: temperature must be positive")

        eps = 1e-12

        # 1. Softmax relaxation: p_o = exp(alpha_o / tau) / sum exp(alpha / tau)
        scaled = [x / temperature for x in arch_alpha_logits]
        max_s = max(scaled)
        exps = [math.exp(x - max_s) for x in scaled]
        sum_exp = sum(exps)
        probs = [e / sum_exp for e in exps]

        # 2. Continuous mixture output: y = sum_o p_o * op_output_o
        mixed_y = [0.0] * D
        for o in range(K):
            p = probs[o]
            op_row = candidate_op_outputs[o]
            for d in range(D):
                mixed_y[d] += p * op_row[d]

        # 3. Discretization argmax and entropy H(p)
        best_op = max(range(K), key=lambda idx: arch_alpha_logits[idx])
        entropy = 0.0
        for p in probs:
            if p > eps:
                entropy -= p * math.log(p)

        return {
            "mixed_edge_output": mixed_y,
            "operation_probabilities": probs,
            "winning_op_index": best_op,
            "architecture_entropy": entropy,
        }
