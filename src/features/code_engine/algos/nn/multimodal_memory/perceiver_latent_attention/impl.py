from __future__ import annotations

import math
from typing import Any, Dict, List
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoPerceiverLatentAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-130
      name: NnAlgoPerceiverLatentAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - perceiver
        - latent_bottleneck
        - cross_attention
      inputs:
        type: object
        required:
          - input_array
          - latent_array
        properties:
          input_array:
            type: array
            items:
              type: array
              items:
                type: number
            description: High-dimensional input sequence of shape (M_inputs, d_model).
          latent_array:
            type: array
            items:
              type: array
              items:
                type: number
            description: Small learned latent query bottleneck of shape (N_latents, d_model).
      outputs:
        type: object
        required:
          - updated_latents
          - compression_ratio
        properties:
          updated_latents:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cross-attended latent representations of shape (N_latents, d_model).
          compression_ratio:
            type: number
            description: Sequence length compression ratio (M_inputs / N_latents).
      parameters: {}
      input_assumptions:
        - len(latent_array) <= len(input_array)
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
          M: input_array length
          N: latent_array length
          d: d_model
        time_worst: O(N * M * d)
        time_typical: O(N * M * d)
        space: O(N * d)
      preconditions:
        - len(input.input_array) > 0 and len(input.latent_array) > 0
        - len(input.input_array[0]) == len(input.latent_array[0])
      postconditions:
        - len(output.updated_latents) == len(input.latent_array)
        - len(output.updated_latents[0]) == len(input.latent_array[0])
      certificate: "Output dimension strictly matches latent bottleneck size N_latents"
      compatible_adapters:
        - ADAPTER-PERCEIVER-RESAMPLER
      related_algos:
        - ALGO-NN-101
        - ALGO-NN-133
      references:
        - "https://arxiv.org/abs/2103.03206"
    ---
    """

    @staticmethod
    def forward(
        input_array: List[List[float]],
        latent_array: List[List[float]],
    ) -> Dict[str, Any]:
        if not input_array or not latent_array:
            raise ValueError("Precondition failed: inputs must be non-empty")
        if len(input_array[0]) != len(latent_array[0]):
            raise ValueError("Precondition failed: feature dimensions must match")

        res = NnAlgoScaledDotProductAttention.forward(
            query=latent_array,
            key=input_array,
            value=input_array,
        )

        ratio = float(len(input_array)) / float(len(latent_array))

        return {
            "updated_latents": res["output"],
            "compression_ratio": ratio,
        }
