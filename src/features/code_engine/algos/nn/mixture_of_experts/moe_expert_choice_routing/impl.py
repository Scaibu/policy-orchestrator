from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMoeExpertChoiceRouting:
    """
    ---
    contract:
      algo_id: ALGO-NN-127
      name: NnAlgoMoeExpertChoiceRouting
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - moe
        - expert_choice
        - balanced_routing
      inputs:
        type: object
        required:
          - affinity_matrix
          - tokens_per_expert
        properties:
          affinity_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Routing affinity scores of shape (num_tokens, num_experts).
          tokens_per_expert:
            type: integer
            minimum: 1
            description: Fixed capacity k of tokens chosen by each expert.
      outputs:
        type: object
        required:
          - expert_assignments
          - load_variance
          - unassigned_tokens
        properties:
          expert_assignments:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Selected token indices per expert of shape (num_experts, tokens_per_expert).
          load_variance:
            type: number
            description: Variance in tokens per expert (strictly 0.0 by construction).
          unassigned_tokens:
            type: integer
            description: Number of tokens selected by zero experts.
      parameters: {}
      input_assumptions:
        - tokens_per_expert <= num_tokens
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact top-k assignment per expert"
      uses_model: false
      complexity:
        variables:
          T: num_tokens
          E: num_experts
          k: tokens_per_expert
        time_worst: O(E * T * log(T))
        time_typical: O(E * T)
        space: O(E * k)
      preconditions:
        - len(input.affinity_matrix) >= input.tokens_per_expert
      postconditions:
        - len(output.expert_assignments) == len(input.affinity_matrix[0])
        - output.load_variance == 0.0
      certificate: "Every expert assigned exactly tokens_per_expert tokens"
      compatible_adapters:
        - ADAPTER-EXPERT-CHOICE
      related_algos:
        - ALGO-NN-125
      references:
        - "https://arxiv.org/abs/2202.09368"
    ---
    """

    @staticmethod
    def route(
        affinity_matrix: List[List[float]],
        tokens_per_expert: int,
    ) -> Dict[str, Any]:
        if not affinity_matrix or tokens_per_expert < 1:
            raise ValueError("Precondition failed: invalid inputs")

        T = len(affinity_matrix)
        E = len(affinity_matrix[0])
        if tokens_per_expert > T:
            raise ValueError("Precondition failed: tokens_per_expert > total tokens")

        expert_assignments: List[List[int]] = []
        token_hit_counts = [0] * T

        for e in range(E):
            scores_for_e = [(t, affinity_matrix[t][e]) for t in range(T)]
            scores_for_e.sort(key=lambda item: item[1], reverse=True)
            chosen_tokens = [t for t, _ in scores_for_e[:tokens_per_expert]]
            for t in chosen_tokens:
                token_hit_counts[t] += 1
            expert_assignments.append(chosen_tokens)

        unassigned = sum(1 for c in token_hit_counts if c == 0)

        return {
            "expert_assignments": expert_assignments,
            "load_variance": 0.0,
            "unassigned_tokens": unassigned,
        }
