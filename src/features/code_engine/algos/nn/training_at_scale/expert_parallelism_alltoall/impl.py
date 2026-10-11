from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoExpertParallelismAlltoall:
    """
    ---
    contract:
      algo_id: ALGO-NN-238
      name: NnAlgoExpertParallelismAlltoall
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - expert_parallelism
        - mixture_of_experts
        - all_to_all
        - dispatch_routing
        - load_balancing
      inputs:
        type: object
        required:
          - token_expert_indices
          - expert_to_rank_map
          - num_ranks
        properties:
          token_expert_indices:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Selected top-k expert IDs for each token in batch of shape (batch_size, top_k).
          expert_to_rank_map:
            type: object
            description: Map associating each global expert ID with its hosting device rank.
          num_ranks:
            type: integer
            description: Total number of device ranks in the expert-parallel group.
          token_hidden_dimension:
            type: integer
            default: 128
            description: Hidden dimension of each token activation vector d_model.
          bytes_per_float:
            type: integer
            default: 2
            description: Floating-point precision bytes (2 for BF16/FP16).
      outputs:
        type: object
        required:
          - rank_dispatch_counts
          - communication_volume_bytes
          - load_imbalance_ratio
          - total_dispatched_tokens
          - num_ranks
        properties:
          rank_dispatch_counts:
            type: array
            items:
              type: integer
            description: Number of token instances routed to each destination device rank.
          communication_volume_bytes:
            type: integer
            description: Total bidirectional All-to-All network transfer volume in bytes.
          load_imbalance_ratio:
            type: number
            description: Ratio of maximum rank token load to mean rank token load.
          total_dispatched_tokens:
            type: integer
            description: Cumulative sum of routed tokens across all destination ranks.
          num_ranks:
            type: integer
            description: Expert parallel degree.
      complexity:
        time: O(len(token_expert_indices) * top_k + num_ranks)
        space: O(num_ranks)
      preconditions:
        - len(input.token_expert_indices) > 0
        - input.num_ranks > 0
        - input.token_hidden_dimension > 0
        - input.bytes_per_float > 0
      postconditions:
        - len(output.rank_dispatch_counts) == input.num_ranks
        - output.total_dispatched_tokens >= len(input.token_expert_indices)
        - output.load_imbalance_ratio >= 1.0
        - output.communication_volume_bytes >= 0
      certificate: "Total dispatched tokens equal batch_size * top_k, routed according to expert_to_rank_map"
      compatible_adapters:
        - ADAPTER-MOE-ALLTOALL-DISPATCHER
        - ADAPTER-EXPERT-LOAD-BALANCER
      related_algos:
        - ALGO-NN-233
        - ALGO-NN-234
      references:
        - "https://arxiv.org/abs/2101.03961"
        - "https://arxiv.org/abs/2201.05596"
    ---
    """

    @classmethod
    def forward(
        cls,
        token_expert_indices: List[List[int]],
        expert_to_rank_map: Dict[str, int],  # string keys to support JSON contract
        num_ranks: int,
        token_hidden_dimension: int = 128,
        bytes_per_float: int = 2,
    ) -> Dict[str, Any]:
        num_tokens = len(token_expert_indices)
        if num_tokens == 0:
            raise ValueError("Precondition failed: token_expert_indices cannot be empty")
        if num_ranks <= 0:
            raise ValueError("Precondition failed: num_ranks must be positive")
        if token_hidden_dimension <= 0 or bytes_per_float <= 0:
            raise ValueError("Precondition failed: dimensions and bytes must be positive")

        dispatch_counts = [0] * num_ranks
        total_routed = 0

        for token_experts in token_expert_indices:
            for exp_id in token_experts:
                # Resolve destination rank
                dest_rank = expert_to_rank_map.get(str(exp_id), exp_id % num_ranks)
                if not (0 <= dest_rank < num_ranks):
                    dest_rank = dest_rank % num_ranks
                dispatch_counts[dest_rank] += 1
                total_routed += 1

        # Calculate imbalance ratio
        max_load = max(dispatch_counts)
        mean_load = float(total_routed) / float(num_ranks)
        imbalance_ratio = float(max_load) / max(1e-12, mean_load)

        # All-to-all communication volume:
        # Dispatch phase: each token travels with d_model floats
        # Combine phase: return token travels back with d_model floats
        # Total network bytes = 2 * total_routed * d_model * bytes_per_float
        comm_bytes = 2 * total_routed * token_hidden_dimension * bytes_per_float

        return {
            "rank_dispatch_counts": dispatch_counts,
            "communication_volume_bytes": comm_bytes,
            "load_imbalance_ratio": imbalance_ratio,
            "total_dispatched_tokens": total_routed,
            "num_ranks": num_ranks,
        }
