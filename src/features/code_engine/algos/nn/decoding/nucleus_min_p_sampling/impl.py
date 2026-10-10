from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNucleusMinPSampling:
    """
    ---
    contract:
      algo_id: ALGO-NN-143
      name: NnAlgoNucleusMinPSampling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - decoding
        - top_p
        - min_p
        - nucleus_sampling
      inputs:
        type: object
        required:
          - logits
        properties:
          logits:
            type: array
            items:
              type: number
            description: Unnormalized token logits of length V.
          top_p:
            type: number
            default: 0.9
            minimum: 0.0
            maximum: 1.0
            description: Cumulative probability threshold p in (0, 1].
          min_p:
            type: number
            default: 0.0
            minimum: 0.0
            maximum: 1.0
            description: Min-p scaling threshold relative to top token probability.
      outputs:
        type: object
        required:
          - retained_indices
          - filtered_probabilities
          - selected_token
        properties:
          retained_indices:
            type: array
            items:
              type: integer
            description: Candidate token indices inside the nucleus.
          filtered_probabilities:
            type: array
            items:
              type: number
            description: Renormalized probabilities over the nucleus.
          selected_token:
            type: integer
            description: Sampled token index.
      parameters: {}
      input_assumptions:
        - 0.0 < top_p <= 1.0
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
        time_worst: O(V * log(V))
        time_typical: O(V * log(V))
        space: O(V)
      preconditions:
        - len(input.logits) > 0
        - 0.0 < input.top_p <= 1.0
      postconditions:
        - len(output.retained_indices) > 0
      certificate: "Cumulative mass of nucleus matches top_p within single token boundary"
      compatible_adapters:
        - ADAPTER-NUCLEUS-SAMPLER
      related_algos:
        - ALGO-NN-142
      references:
        - "https://arxiv.org/abs/1904.09751"
        - "https://arxiv.org/abs/2407.01082"
    ---
    """

    @staticmethod
    def sample(
        logits: List[float],
        top_p: float = 0.9,
        min_p: float = 0.0,
    ) -> Dict[str, Any]:
        if not logits or not (0.0 < top_p <= 1.0) or not (0.0 <= min_p <= 1.0):
            raise ValueError("Precondition failed: invalid sampling thresholds")

        V = len(logits)
        max_l = max(logits)
        exp_vals = [math.exp(v - max_l) for v in logits]
        sum_exp = sum(exp_vals)
        base_probs = [e / sum_exp for e in exp_vals]

        indexed = list(enumerate(base_probs))
        indexed.sort(key=lambda item: item[1], reverse=True)

        top_prob = indexed[0][1]
        threshold = min_p * top_prob

        cum_sum = 0.0
        retained: List[int] = []
        for idx, prob in indexed:
            if prob < threshold and len(retained) > 0:
                break
            retained.append(idx)
            cum_sum += prob
            if cum_sum >= top_p:
                break

        retained_probs = [base_probs[idx] for idx in retained]
        total_p = sum(retained_probs)
        norm_probs = [p / total_p for p in retained_probs]

        return {
            "retained_indices": retained,
            "filtered_probabilities": norm_probs,
            "selected_token": retained[0],
        }
