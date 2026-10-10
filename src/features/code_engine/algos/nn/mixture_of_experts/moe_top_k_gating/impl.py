from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class NnAlgoMoeTopKGating:
    """
    ---
    contract:
      algo_id: ALGO-NN-125
      name: NnAlgoMoeTopKGating
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - moe
        - top_k_gating
        - sparse_mixture_of_experts
      inputs:
        type: object
        required:
          - router_logits
          - top_k
        properties:
          router_logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Router unnormalized scores of shape (num_tokens, num_experts).
          top_k:
            type: integer
            minimum: 1
            description: Number of active experts per token.
      outputs:
        type: object
        required:
          - selected_experts
          - gating_weights
          - dispatch_mask
        properties:
          selected_experts:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Selected top-k expert indices per token of shape (num_tokens, top_k).
          gating_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Normalized softmax routing weights of shape (num_tokens, top_k).
          dispatch_mask:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Binary mask of shape (num_tokens, num_experts).
      parameters: {}
      input_assumptions:
        - 1 <= top_k <= num_experts
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point bounds"
      uses_model: false
      complexity:
        variables:
          T: num_tokens
          E: num_experts
          k: top_k
        time_worst: O(T * E + T * k * log(E))
        time_typical: O(T * E)
        space: O(T * E)
      preconditions:
        - len(input.router_logits) > 0
        - 1 <= input.top_k <= len(input.router_logits[0])
      postconditions:
        - len(output.selected_experts) == len(input.router_logits)
        - len(output.selected_experts[0]) == input.top_k
      certificate: "Per-token gating weights sum to 1.0 within numerical precision"
      compatible_adapters:
        - ADAPTER-MOE-GATING
      related_algos:
        - ALGO-NN-126
        - ALGO-NN-127
      references:
        - "https://arxiv.org/abs/1701.06538"
        - "https://arxiv.org/abs/2101.03961"
    ---
    """

    @staticmethod
    def route(
        router_logits: List[List[float]],
        top_k: int = 2,
    ) -> Dict[str, Any]:
        if not router_logits:
            raise ValueError("Precondition failed: router_logits must be non-empty")
        num_experts = len(router_logits[0])
        if not (1 <= top_k <= num_experts):
            raise ValueError("Precondition failed: 1 <= top_k <= num_experts")

        selected_experts: List[List[int]] = []
        gating_weights: List[List[float]] = []
        dispatch_mask: List[List[int]] = []

        for row in router_logits:
            indexed = list(enumerate(row))
            indexed.sort(key=lambda item: item[1], reverse=True)
            top_items = indexed[:top_k]

            top_indices = [idx for idx, _ in top_items]
            top_scores = [score for _, score in top_items]

            max_score = max(top_scores)
            exp_scores = [math.exp(s - max_score) for s in top_scores]
            sum_exp = sum(exp_scores)
            norm_weights = [e / sum_exp for e in exp_scores] if sum_exp > 0 else [1.0 / top_k] * top_k

            mask_row = [0] * num_experts
            for idx in top_indices:
                mask_row[idx] = 1

            selected_experts.append(top_indices)
            gating_weights.append(norm_weights)
            dispatch_mask.append(mask_row)

        return {
            "selected_experts": selected_experts,
            "gating_weights": gating_weights,
            "dispatch_mask": dispatch_mask,
        }
