from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoTopKSampling:
    """
    ---
    contract:
      algo_id: ALGO-NN-142
      name: NnAlgoTopKSampling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - decoding
        - top_k
        - truncation
      inputs:
        type: object
        required:
          - logits
          - k
        properties:
          logits:
            type: array
            items:
              type: number
            description: Raw token logits of length V.
          k:
            type: integer
            minimum: 1
            description: Cutoff rank k.
      outputs:
        type: object
        required:
          - filtered_probabilities
          - retained_indices
          - selected_token
        properties:
          filtered_probabilities:
            type: array
            items:
              type: number
            description: Renormalized probabilities across top-k tokens.
          retained_indices:
            type: array
            items:
              type: integer
            description: Indices of top-k tokens.
          selected_token:
            type: integer
            description: Sampled token index.
      parameters: {}
      input_assumptions:
        - 1 <= k <= len(logits)
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
          V: vocab_size
          k: k cutoff
        time_worst: O(V * log(V))
        time_typical: O(V + k * log(V))
        space: O(V)
      preconditions:
        - len(input.logits) > 0
        - 1 <= input.k <= len(input.logits)
      postconditions:
        - len(output.retained_indices) == input.k
      certificate: "Filtered probabilities sum to 1.0 within numerical precision"
      compatible_adapters:
        - ADAPTER-TOP-K-SAMPLER
      related_algos:
        - ALGO-NN-141
        - ALGO-NN-143
      references:
        - "https://arxiv.org/abs/1805.04833"
    ---
    """

    @staticmethod
    def sample(logits: List[float], k: int) -> Dict[str, Any]:
        if not logits or k < 1 or k > len(logits):
            raise ValueError("Precondition failed: invalid k parameter")

        indexed = list(enumerate(logits))
        indexed.sort(key=lambda item: item[1], reverse=True)
        top_k_items = indexed[:k]

        top_indices = [idx for idx, _ in top_k_items]
        top_logits = [val for _, val in top_k_items]

        max_l = max(top_logits)
        exp_vals = [math.exp(v - max_l) for v in top_logits]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        best_token = top_indices[0]

        return {
            "filtered_probabilities": probs,
            "retained_indices": top_indices,
            "selected_token": best_token,
        }
