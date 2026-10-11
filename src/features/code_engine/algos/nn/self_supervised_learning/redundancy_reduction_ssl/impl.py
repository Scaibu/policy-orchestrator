from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRedundancyReductionSsl:
    """
    ---
    contract:
      algo_id: ALGO-NN-173
      name: NnAlgoRedundancyReductionSsl
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_supervised_learning
        - barlow_twins
        - vicreg
        - redundancy_reduction
        - decorrelation
      inputs:
        type: object
        required:
          - batch_z1
          - batch_z2
        properties:
          batch_z1:
            type: array
            items:
              type: array
              items:
                type: number
            description: Embeddings of view 1 of shape (B, D).
          batch_z2:
            type: array
            items:
              type: array
              items:
                type: number
            description: Embeddings of view 2 of shape (B, D).
          lambda_off_diagonal:
            type: number
            default: 0.005
            description: Barlow Twins off-diagonal redundancy penalty multiplier.
      outputs:
        type: object
        required:
          - barlow_loss
          - on_diagonal_invariance_loss
          - off_diagonal_redundancy_loss
          - mean_cross_correlation_diag
        properties:
          barlow_loss:
            type: number
            description: Total Barlow Twins loss = invariance + lambda * redundancy.
          on_diagonal_invariance_loss:
            type: number
            description: Sum of (1 - C_ii)^2 over feature dimensions.
          off_diagonal_redundancy_loss:
            type: number
            description: Sum of C_ij^2 for i != j.
          mean_cross_correlation_diag:
            type: number
            description: Average diagonal cross-correlation.
      parameters: {}
      input_assumptions:
        - batch_z1 and batch_z2 have matching dimensions B >= 2 and D >= 1
        - lambda_off_diagonal >= 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized batch standard deviations with eps 1e-8"
      uses_model: false
      complexity:
        variables:
          B: batch size
          D: embedding dimension
        time_worst: O(B * D + D^2)
        time_typical: O(B * D + D^2)
        space: O(B * D + D^2)
      preconditions:
        - len(input.batch_z1) >= 2 and len(input.batch_z1[0]) >= 1
        - len(input.batch_z1) == len(input.batch_z2)
        - len(input.batch_z1[0]) == len(input.batch_z2[0])
      postconditions:
        - output.barlow_loss >= 0.0
        - output.on_diagonal_invariance_loss >= 0.0
        - output.off_diagonal_redundancy_loss >= 0.0
      certificate: "Barlow loss strictly equals on_diag + lambda * off_diag"
      compatible_adapters:
        - ADAPTER-BARLOW-TWINS
        - ADAPTER-VICREG-OBJECTIVE
      related_algos:
        - ALGO-NN-172
        - ALGO-NN-175
      references:
        - "https://arxiv.org/abs/2103.03230"
        - "https://arxiv.org/abs/2105.04906"
    ---
    """

    @classmethod
    def forward(
        cls,
        batch_z1: List[List[float]],
        batch_z2: List[List[float]],
        lambda_off_diagonal: float = 0.005,
    ) -> Dict[str, Any]:
        B = len(batch_z1)
        if B < 2 or len(batch_z1[0]) == 0:
            raise ValueError("Precondition failed: batch size must be >= 2 and dimensions non-empty")
        D = len(batch_z1[0])

        if len(batch_z2) != B or any(len(r) != D for r in batch_z1) or any(len(r) != D for r in batch_z2):
            raise ValueError("Precondition failed: uniform matrix dimensions required")
        if lambda_off_diagonal < 0.0:
            raise ValueError("Precondition failed: lambda_off_diagonal must be non-negative")

        eps = 1e-8

        # 1. Standardize along batch dimension for each feature column:
        # z_norm = (z - mean) / std
        norm_z1: List[List[float]] = [[0.0] * D for _ in range(B)]
        norm_z2: List[List[float]] = [[0.0] * D for _ in range(B)]

        for j in range(D):
            mean1 = sum(batch_z1[i][j] for i in range(B)) / float(B)
            var1 = sum((batch_z1[i][j] - mean1) ** 2 for i in range(B)) / float(B)
            std1 = math.sqrt(var1) + eps

            mean2 = sum(batch_z2[i][j] for i in range(B)) / float(B)
            var2 = sum((batch_z2[i][j] - mean2) ** 2 for i in range(B)) / float(B)
            std2 = math.sqrt(var2) + eps

            for i in range(B):
                norm_z1[i][j] = (batch_z1[i][j] - mean1) / std1
                norm_z2[i][j] = (batch_z2[i][j] - mean2) / std2

        # 2. Compute cross-correlation matrix C: C_ij = (1 / B) * sum_{b} z1[b, i] * z2[b, j]
        on_diag_loss = 0.0
        off_diag_loss = 0.0
        diag_sum = 0.0

        for i in range(D):
            for j in range(D):
                c_ij = sum(norm_z1[b][i] * norm_z2[b][j] for b in range(B)) / float(B)
                if i == j:
                    diff = 1.0 - c_ij
                    on_diag_loss += diff * diff
                    diag_sum += c_ij
                else:
                    off_diag_loss += c_ij * c_ij

        total_loss = on_diag_loss + lambda_off_diagonal * off_diag_loss
        mean_diag = diag_sum / float(D)

        return {
            "barlow_loss": total_loss,
            "on_diagonal_invariance_loss": on_diag_loss,
            "off_diagonal_redundancy_loss": off_diag_loss,
            "mean_cross_correlation_diag": mean_diag,
        }
