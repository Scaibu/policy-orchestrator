from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMagnitudePruningLotteryTicket:
    """
    ---
    contract:
      algo_id: ALGO-NN-248
      name: NnAlgoMagnitudePruningLotteryTicket
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - pruning_sparsity
        - magnitude_pruning
        - lottery_ticket_hypothesis
        - iterative_pruning
        - unstructured_sparsity
      inputs:
        type: object
        required:
          - current_weights
          - initial_weights
        properties:
          current_weights:
            type: array
            items:
              type: number
            description: Trained model parameter vector theta_t of dimension P.
          initial_weights:
            type: array
            items:
              type: number
            description: Early checkpoint initialization weights theta_0 for lottery ticket rewinding.
          target_sparsity:
            type: number
            default: 0.5
            description: Desired fraction of weights to be zeroed out in [0.0, 1.0).
          rewind_weights:
            type: boolean
            default: true
            description: If True, reset surviving weights to initial_weights (Lottery Ticket); if False, keep current_weights.
      outputs:
        type: object
        required:
          - pruned_weights
          - binary_mask
          - achieved_sparsity
          - pruning_threshold
          - surviving_parameter_count
        properties:
          pruned_weights:
            type: array
            items:
              type: number
            description: Parameter vector with lowest magnitude entries zeroed out.
          binary_mask:
            type: array
            items:
              type: integer
            description: Binary mask m in {0, 1}^P indicating surviving weights.
          achieved_sparsity:
            type: number
            description: Exact fraction of weights set to 0.0.
          pruning_threshold:
            type: number
            description: Absolute magnitude cutoff value tau below which weights are pruned.
          surviving_parameter_count:
            type: integer
            description: Number of nonzero parameters preserved in the winning ticket.
      complexity:
        time: O(P log P)
        space: O(P)
      preconditions:
        - len(input.current_weights) > 0
        - len(input.current_weights) == len(input.initial_weights)
        - 0.0 <= input.target_sparsity < 1.0
      postconditions:
        - len(output.pruned_weights) == len(input.current_weights)
        - len(output.binary_mask) == len(input.current_weights)
        - 0.0 <= output.achieved_sparsity <= 1.0
        - output.surviving_parameter_count <= len(input.current_weights)
      certificate: "Pruned weights retain exactly the top (1 - target_sparsity) fraction by absolute magnitude"
      compatible_adapters:
        - ADAPTER-MAGNITUDE-PRUNER
        - ADAPTER-LOTTERY-TICKET-REWINDER
      related_algos:
        - ALGO-NN-249
        - ALGO-NN-250
      references:
        - "https://arxiv.org/abs/1803.03635"
        - "https://arxiv.org/abs/1903.01611"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_weights: List[float],
        initial_weights: List[float],
        target_sparsity: float = 0.5,
        rewind_weights: bool = True,
    ) -> Dict[str, Any]:
        p = len(current_weights)
        if p == 0:
            raise ValueError("Precondition failed: current_weights cannot be empty")
        if len(initial_weights) != p:
            raise ValueError("Precondition failed: initial_weights length must match current_weights")
        if not (0.0 <= target_sparsity < 1.0):
            raise ValueError("Precondition failed: target_sparsity must be in [0.0, 1.0)")

        if target_sparsity == 0.0:
            return {
                "pruned_weights": list(initial_weights if rewind_weights else current_weights),
                "binary_mask": [1] * p,
                "achieved_sparsity": 0.0,
                "pruning_threshold": 0.0,
                "surviving_parameter_count": p,
            }

        # Number of weights to prune
        num_prune = int(math.floor(target_sparsity * p))
        num_prune = max(0, min(p - 1, num_prune))

        # Sort indices by absolute weight magnitude ascending
        sorted_indices = sorted(range(p), key=lambda idx: abs(current_weights[idx]))
        pruned_indices_set = set(sorted_indices[:num_prune])
        threshold = abs(current_weights[sorted_indices[num_prune]]) if num_prune < p else 0.0

        mask: List[int] = []
        out_weights: List[float] = []
        surviving_count = 0

        for i in range(p):
            if i in pruned_indices_set:
                mask.append(0)
                out_weights.append(0.0)
            else:
                mask.append(1)
                val = initial_weights[i] if rewind_weights else current_weights[i]
                out_weights.append(val)
                surviving_count += 1

        achieved_sparsity = float(p - surviving_count) / float(p)

        return {
            "pruned_weights": out_weights,
            "binary_mask": mask,
            "achieved_sparsity": achieved_sparsity,
            "pruning_threshold": threshold,
            "surviving_parameter_count": surviving_count,
        }
