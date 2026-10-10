from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSelfConsistencySampling:
    """
    ---
    contract:
      algo_id: ALGO-NN-149
      name: NnAlgoSelfConsistencySampling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_consistency
        - majority_voting
        - ensembling
      inputs:
        type: object
        required:
          - sampled_answers
        properties:
          sampled_answers:
            type: array
            items:
              type: string
            description: List of final extracted answers from N parallel generation paths.
      outputs:
        type: object
        required:
          - consensus_answer
          - agreement_rate
          - vote_distribution
        properties:
          consensus_answer:
            type: string
            description: Plurality majority voted answer.
          agreement_rate:
            type: number
            description: Fraction of sampled paths agreeing on consensus answer.
          vote_distribution:
            type: object
            description: Frequency table of answer occurrences.
      parameters: {}
      input_assumptions:
        - sampled_answers is non-empty
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact discrete voting"
      uses_model: false
      complexity:
        variables:
          N: num_samples
        time_worst: O(N)
        time_typical: O(N)
        space: O(N)
      preconditions:
        - len(input.sampled_answers) > 0
      postconditions:
        - output.agreement_rate >= (1.0 / len(input.sampled_answers))
      certificate: "consensus_answer is the mode of sampled_answers"
      compatible_adapters:
        - ADAPTER-SELF-CONSISTENCY
      related_algos:
        - ALGO-NN-141
      references:
        - "https://arxiv.org/abs/2203.11171"
    ---
    """

    @staticmethod
    def aggregate_votes(sampled_answers: List[str]) -> Dict[str, Any]:
        if not sampled_answers:
            raise ValueError("Precondition failed: sampled_answers must be non-empty")

        N = len(sampled_answers)
        freq: Dict[str, int] = {}
        for ans in sampled_answers:
            norm_ans = ans.strip()
            freq[norm_ans] = freq.get(norm_ans, 0) + 1

        best_ans = max(freq.keys(), key=lambda k: freq[k])
        rate = float(freq[best_ans]) / float(N)

        return {
            "consensus_answer": best_ans,
            "agreement_rate": rate,
            "vote_distribution": freq,
        }
