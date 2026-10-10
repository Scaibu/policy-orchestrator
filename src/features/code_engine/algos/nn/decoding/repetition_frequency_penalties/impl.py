from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRepetitionFrequencyPenalties:
    """
    ---
    contract:
      algo_id: ALGO-NN-144
      name: NnAlgoRepetitionFrequencyPenalties
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - decoding
        - repetition_penalty
        - frequency_penalty
      inputs:
        type: object
        required:
          - logits
          - past_tokens
        properties:
          logits:
            type: array
            items:
              type: number
            description: Raw next-token logits of length V.
          past_tokens:
            type: array
            items:
              type: integer
            description: List of already generated token IDs.
          repetition_penalty:
            type: number
            default: 1.1
            description: Multiplicative penalty (> 1.0).
          frequency_penalty:
            type: number
            default: 0.0
            description: Subtractive count penalty alpha.
          presence_penalty:
            type: number
            default: 0.0
            description: Subtractive existence penalty beta.
      outputs:
        type: object
        required:
          - adjusted_logits
          - penalized_tokens_count
        properties:
          adjusted_logits:
            type: array
            items:
              type: number
            description: Adjusted logits after applying penalties.
          penalized_tokens_count:
            type: integer
            description: Number of unique tokens modified.
      parameters: {}
      input_assumptions:
        - repetition_penalty >= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact arithmetic penalty"
      uses_model: false
      complexity:
        variables:
          V: vocab_size
          T: past_tokens length
        time_worst: O(V + T)
        time_typical: O(V + T)
        space: O(V)
      preconditions:
        - len(input.logits) > 0
        - input.repetition_penalty >= 1.0
      postconditions:
        - len(output.adjusted_logits) == len(input.logits)
      certificate: "Logits for seen tokens decreased monotonically"
      compatible_adapters:
        - ADAPTER-LOGIT-PENALIZER
      related_algos:
        - ALGO-NN-141
      references:
        - "https://arxiv.org/abs/1909.05858"
    ---
    """

    @staticmethod
    def apply(
        logits: List[float],
        past_tokens: List[int],
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        if not logits or repetition_penalty < 1.0:
            raise ValueError("Precondition failed: invalid penalty parameters")

        adjusted = logits[:]
        counts: Dict[int, int] = {}
        for tok in past_tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, cnt in counts.items():
            if 0 <= tok < len(adjusted):
                val = adjusted[tok]
                if val > 0:
                    val /= repetition_penalty
                else:
                    val *= repetition_penalty

                val -= frequency_penalty * float(cnt)
                val -= presence_penalty * (1.0 if cnt > 0 else 0.0)
                adjusted[tok] = val

        return {
            "adjusted_logits": adjusted,
            "penalized_tokens_count": len(counts),
        }
