from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoStructuredOneshotPruningLlms:
    """
    ---
    contract:
      algo_id: ALGO-NN-249
      name: NnAlgoStructuredOneshotPruningLlms
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - pruning_sparsity
        - structured_pruning
        - wanda
        - oneshot_pruning
        - activation_awareness
      inputs:
        type: object
        required:
          - weight_matrix
          - activation_norms
        properties:
          weight_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: 2D linear weight matrix W of shape (d_out, d_in).
          activation_norms:
            type: array
            items:
              type: number
            description: L2 norm of input activations ||X_{:, j}||_2 from calibration data of dimension d_in.
          pruning_ratio:
            type: number
            default: 0.5
            description: Target sparsity ratio per output row in [0.0, 1.0).
          mode:
            type: string
            default: wanda
            enum: [wanda, magnitude]
            description: Importance metric: wanda (|W_{ij}| * ||X_j||_2) vs magnitude (|W_{ij}|).
      outputs:
        type: object
        required:
          - pruned_matrix
          - importance_scores
          - achieved_sparsity
          - zeroed_parameter_count
          - total_parameters
        properties:
          pruned_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Sparsified weight matrix of shape (d_out, d_in).
          importance_scores:
            type: array
            items:
              type: array
              items:
                type: number
            description: Computed importance score S_{ij} for every entry.
          achieved_sparsity:
            type: number
            description: Actual fraction of parameters set to zero across the matrix.
          zeroed_parameter_count:
            type: integer
            description: Total count of pruned zero weights.
          total_parameters:
            type: integer
            description: Total matrix elements d_out * d_in.
      complexity:
        time: O(d_out * d_in * log(d_in))
        space: O(d_out * d_in)
      preconditions:
        - len(input.weight_matrix) > 0
        - len(input.activation_norms) == len(input.weight_matrix[0])
        - all(len(row) == len(input.activation_norms) for row in input.weight_matrix)
        - 0.0 <= input.pruning_ratio < 1.0
      postconditions:
        - len(output.pruned_matrix) == len(input.weight_matrix)
        - all(len(row) == len(input.activation_norms) for row in output.pruned_matrix)
        - 0.0 <= output.achieved_sparsity < 1.0
      certificate: "Prunes lowest pruning_ratio fraction of entries per row according to Wanda importance"
      compatible_adapters:
        - ADAPTER-WANDA-ONESHOT-PRUNER
        - ADAPTER-SPARSEGPT-STRUCTURED-MASKER
      related_algos:
        - ALGO-NN-248
        - ALGO-NN-250
      references:
        - "https://arxiv.org/abs/2306.11695"
        - "https://arxiv.org/abs/2301.00774"
    ---
    """

    @classmethod
    def forward(
        cls,
        weight_matrix: List[List[float]],
        activation_norms: List[float],
        pruning_ratio: float = 0.5,
        mode: str = "wanda",
    ) -> Dict[str, Any]:
        d_out = len(weight_matrix)
        if d_out == 0:
            raise ValueError("Precondition failed: weight_matrix cannot be empty")
        d_in = len(activation_norms)
        if d_in == 0 or any(len(row) != d_in for row in weight_matrix):
            raise ValueError("Precondition failed: row length must match activation_norms length")
        if not (0.0 <= pruning_ratio < 1.0):
            raise ValueError("Precondition failed: pruning_ratio must be in [0.0, 1.0)")

        num_prune_per_row = int(math.floor(pruning_ratio * d_in))

        pruned_matrix: List[List[float]] = []
        importance_scores: List[List[float]] = []
        total_zeroed = 0

        for i in range(d_out):
            row_w = weight_matrix[i]
            row_scores: List[float] = []

            for j in range(d_in):
                if mode == "magnitude":
                    score = abs(row_w[j])
                else:
                    # Wanda: |W_ij| * ||X_j||_2
                    score = abs(row_w[j]) * activation_norms[j]
                row_scores.append(score)

            importance_scores.append(row_scores)

            if num_prune_per_row > 0:
                # Identify lowest score indices in this row
                sorted_col_indices = sorted(range(d_in), key=lambda idx: row_scores[idx])
                pruned_cols_set = set(sorted_col_indices[:num_prune_per_row])

                new_row = [0.0 if j in pruned_cols_set else row_w[j] for j in range(d_in)]
                total_zeroed += num_prune_per_row
            else:
                new_row = list(row_w)

            pruned_matrix.append(new_row)

        total_params = d_out * d_in
        achieved_sparsity = float(total_zeroed) / float(total_params)

        return {
            "pruned_matrix": pruned_matrix,
            "importance_scores": importance_scores,
            "achieved_sparsity": achieved_sparsity,
            "zeroed_parameter_count": total_zeroed,
            "total_parameters": total_params,
        }
