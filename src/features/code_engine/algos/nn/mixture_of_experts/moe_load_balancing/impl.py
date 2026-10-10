from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMoeLoadBalancing:
    """
    ---
    contract:
      algo_id: ALGO-NN-126
      name: NnAlgoMoeLoadBalancing
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - moe
        - load_balancing
        - auxiliary_loss
      inputs:
        type: object
        required:
          - router_probs
          - capacity_factor
          - num_experts
        properties:
          router_probs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Softmax router probabilities of shape (num_tokens, num_experts).
          capacity_factor:
            type: number
            default: 1.25
            description: Expert capacity scaling factor C.
          num_experts:
            type: integer
            minimum: 1
            description: Total expert count E.
      outputs:
        type: object
        required:
          - aux_loss
          - expert_token_counts
          - dropped_tokens_count
          - token_drop_rate
        properties:
          aux_loss:
            type: number
            description: Scaled load balancing auxiliary loss.
          expert_token_counts:
            type: array
            items:
              type: integer
            description: Number of tokens assigned to each expert.
          dropped_tokens_count:
            type: integer
            description: Count of tokens dropped due to expert capacity saturation.
          token_drop_rate:
            type: number
            description: Percentage of dropped tokens.
      parameters: {}
      input_assumptions:
        - capacity_factor >= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical bounds"
      uses_model: false
      complexity:
        variables:
          T: num_tokens
          E: num_experts
        time_worst: O(T * E)
        time_typical: O(T * E)
        space: O(E)
      preconditions:
        - len(input.router_probs) > 0
        - input.capacity_factor >= 1.0
      postconditions:
        - output.aux_loss >= 0.0
        - 0.0 <= output.token_drop_rate <= 1.0
      certificate: "aux_loss == num_experts * sum_e (f_e * P_e)"
      compatible_adapters:
        - ADAPTER-MOE-LOSS
      related_algos:
        - ALGO-NN-125
        - ALGO-NN-127
      references:
        - "https://arxiv.org/abs/2101.03961"
    ---
    """

    @staticmethod
    def compute_loss_and_capacity(
        router_probs: List[List[float]],
        capacity_factor: float = 1.25,
        num_experts: int = 8,
    ) -> Dict[str, Any]:
        if not router_probs or capacity_factor < 1.0 or num_experts < 1:
            raise ValueError("Precondition failed: invalid inputs")

        T = len(router_probs)
        expert_capacity = int(math.ceil((float(T) / float(num_experts)) * capacity_factor))

        expert_counts = [0] * num_experts
        dropped = 0

        for row in router_probs:
            best_e = max(range(num_experts), key=lambda idx: row[idx])
            if expert_counts[best_e] < expert_capacity:
                expert_counts[best_e] += 1
            else:
                dropped += 1

        f_e = [float(c) / float(T) for c in expert_counts]
        P_e = [0.0] * num_experts
        for row in router_probs:
            for e in range(num_experts):
                P_e[e] += row[e] / float(T)

        aux_loss = float(num_experts) * sum(f_e[e] * P_e[e] for e in range(num_experts))
        drop_rate = float(dropped) / float(T)

        return {
            "aux_loss": aux_loss,
            "expert_token_counts": expert_counts,
            "dropped_tokens_count": dropped,
            "token_drop_rate": drop_rate,
        }
