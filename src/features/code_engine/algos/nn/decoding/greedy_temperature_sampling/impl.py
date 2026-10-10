from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoGreedyTemperatureSampling:
    """
    ---
    contract:
      algo_id: ALGO-NN-141
      name: NnAlgoGreedyTemperatureSampling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - decoding
        - temperature_sampling
        - greedy_search
      inputs:
        type: object
        required:
          - logits
        properties:
          logits:
            type: array
            items:
              type: number
            description: 1D next-token logits of shape (vocab_size).
          temperature:
            type: number
            default: 1.0
            minimum: 0.0
            description: Sampling temperature T (0.0 implies deterministic greedy argmax).
      outputs:
        type: object
        required:
          - selected_token
          - probabilities
          - is_greedy
        properties:
          selected_token:
            type: integer
            description: Chosen token index.
          probabilities:
            type: array
            items:
              type: number
            description: Temperature-scaled softmax probability distribution.
          is_greedy:
            type: boolean
            description: True if greedy decoding was invoked.
      parameters: {}
      input_assumptions:
        - logits is non-empty 1D array of finite numbers
        - temperature >= 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point precision"
      uses_model: false
      complexity:
        variables:
          V: vocab_size
        time_worst: O(V)
        time_typical: O(V)
        space: O(V)
      preconditions:
        - len(input.logits) > 0
        - input.temperature >= 0.0
      postconditions:
        - 0 <= output.selected_token < len(input.logits)
      certificate: "When temperature == 0.0, output is argmax(logits)"
      compatible_adapters:
        - ADAPTER-DECODING-SAMPLER
      related_algos:
        - ALGO-NN-142
        - ALGO-NN-143
      references:
        - "https://doi.org/10.1162/neco.1989.1.4.532"
    ---
    """

    @staticmethod
    def sample(logits: List[float], temperature: float = 1.0) -> Dict[str, Any]:
        if not logits or temperature < 0.0:
            raise ValueError("Precondition failed: logits non-empty and temperature >= 0.0")

        V = len(logits)
        if temperature == 0.0 or temperature < 1e-6:
            best_idx = max(range(V), key=lambda i: logits[i])
            probs = [1.0 if i == best_idx else 0.0 for i in range(V)]
            return {
                "selected_token": best_idx,
                "probabilities": probs,
                "is_greedy": True,
            }

        scaled = [v / temperature for v in logits]
        max_l = max(scaled)
        exp_vals = [math.exp(v - max_l) for v in scaled]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        best_idx = max(range(V), key=lambda i: probs[i])

        return {
            "selected_token": best_idx,
            "probabilities": probs,
            "is_greedy": False,
        }
