from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPatchTimeSeriesTransformer:
    """
    ---
    contract:
      algo_id: ALGO-NN-190
      name: NnAlgoPatchTimeSeriesTransformer
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - time_series
        - forecasting
        - transformer
        - patchtst
        - revin
        - tokenization
      inputs:
        type: object
        required:
          - time_series
          - patch_len
          - stride
          - projection_weights
        properties:
          time_series:
            type: array
            items:
              type: number
            description: Univariate raw continuous time series of length L.
          patch_len:
            type: integer
            minimum: 1
            description: Patch window length P (number of consecutive timesteps per token).
          stride:
            type: integer
            minimum: 1
            description: Stride S between consecutive patches.
          projection_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Patch linear embedding projection matrix W_proj of shape (D_embed, P).
      outputs:
        type: object
        required:
          - normalized_series
          - patch_tokens
          - projected_embeddings
          - instance_mean
          - instance_std
        properties:
          normalized_series:
            type: array
            items:
              type: number
            description: RevIN zero-mean unit-variance normalized series of length L.
          patch_tokens:
            type: array
            items:
              type: array
              items:
                type: number
            description: Unfolded patches of shape (N_patches, P).
          projected_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Embedded transformer token sequence of shape (N_patches, D_embed).
          instance_mean:
            type: number
            description: RevIN instance mean scalar.
          instance_std:
            type: number
            description: RevIN instance standard deviation scalar.
      complexity:
        time: O(L + N_patches * D_embed * P)
        space: O(L + N_patches * (P + D_embed))
      preconditions:
        - len(input.time_series) >= input.patch_len
        - input.patch_len >= 1 and input.stride >= 1
        - len(input.projection_weights) > 0 and len(input.projection_weights[0]) == input.patch_len
      postconditions:
        - len(output.normalized_series) == len(input.time_series)
        - len(output.patch_tokens) == len(output.projected_embeddings)
        - output.instance_std > 0.0
      certificate: "Each patch token has length exactly equal to patch_len"
      compatible_adapters:
        - ADAPTER-PATCHTST-TOKENIZER
        - ADAPTER-REVIN-NORMALIZER
      related_algos:
        - ALGO-NN-189
      references:
        - "https://arxiv.org/abs/2211.14730"
    ---
    """

    @classmethod
    def forward(
        cls,
        time_series: List[float],
        patch_len: int,
        stride: int,
        projection_weights: List[List[float]],
    ) -> Dict[str, Any]:
        L = len(time_series)
        if L < patch_len or patch_len < 1 or stride < 1:
            raise ValueError("Precondition failed: invalid series length, patch_len, or stride")

        D_embed = len(projection_weights)
        if D_embed == 0 or any(len(r) != patch_len for r in projection_weights):
            raise ValueError("Precondition failed: projection_weights dimension mismatch with patch_len")

        eps = 1e-8

        # 1. Reversible Instance Normalization (RevIN)
        mean_val = sum(time_series) / float(L)
        var_val = sum((x - mean_val) ** 2 for x in time_series) / float(L)
        std_val = math.sqrt(var_val) + eps

        norm_series = [(x - mean_val) / std_val for x in time_series]

        # 2. Patch Extraction (sliding window over normalized series)
        patches: List[List[float]] = []
        start_idx = 0
        while start_idx + patch_len <= L:
            patch = norm_series[start_idx : start_idx + patch_len]
            patches.append(patch)
            start_idx += stride

        if len(patches) == 0:
            raise ValueError("Precondition failed: no patches could be formed")

        # 3. Linear Token Projection: token_i = W_proj * patch_i
        projected_tokens: List[List[float]] = []
        for patch in patches:
            token_embed: List[float] = []
            for d in range(D_embed):
                val = sum(projection_weights[d][p] * patch[p] for p in range(patch_len))
                token_embed.append(val)
            projected_tokens.append(token_embed)

        return {
            "normalized_series": norm_series,
            "patch_tokens": patches,
            "projected_embeddings": projected_tokens,
            "instance_mean": mean_val,
            "instance_std": std_val,
        }
