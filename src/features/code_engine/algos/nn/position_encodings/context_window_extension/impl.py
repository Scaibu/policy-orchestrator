from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoContextWindowExtension:
    """
    ---
    contract:
      algo_id: ALGO-NN-111
      name: NnAlgoContextWindowExtension
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - context_extension
        - position_interpolation
        - ntk_scaling
        - yarn
      inputs:
        type: object
        required:
          - scale_factor
          - method
          - original_max_len
        properties:
          scale_factor:
            type: number
            minimum: 1.0
            description: Context expansion scale factor s >= 1.0.
          method:
            type: string
            enum: ["linear_interpolation", "ntk_aware", "yarn"]
            description: Scaling methodology.
          original_max_len:
            type: integer
            minimum: 1
            description: Pretrained context length L_orig.
          d_model:
            type: integer
            default: 128
            description: Head dimension.
          base:
            type: number
            default: 10000.0
            description: RoPE base frequency.
      outputs:
        type: object
        required:
          - extended_max_len
          - scaled_frequencies
          - effective_scale
        properties:
          extended_max_len:
            type: integer
            description: Extended context window size.
          scaled_frequencies:
            type: array
            items:
              type: number
            description: Scaled theta frequencies per dimension pair.
          effective_scale:
            type: number
            description: Final scaling ratio.
      parameters: {}
      input_assumptions:
        - scale_factor >= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact scaling formulas"
      uses_model: false
      complexity:
        variables:
          d: d_model
        time_worst: O(d)
        time_typical: O(d)
        space: O(d)
      preconditions:
        - input.scale_factor >= 1.0
        - input.original_max_len >= 1
      postconditions:
        - output.extended_max_len >= input.original_max_len
      certificate: "extended_max_len == int(scale_factor * original_max_len)"
      compatible_adapters:
        - ADAPTER-ROPE-SCALING
      related_algos:
        - ALGO-NN-108
      references:
        - "https://arxiv.org/abs/2306.15595"
        - "https://arxiv.org/abs/2309.00071"
    ---
    """

    @staticmethod
    def forward(
        scale_factor: float,
        method: str,
        original_max_len: int,
        d_model: int = 128,
        base: float = 10000.0,
    ) -> Dict[str, Any]:
        if scale_factor < 1.0 or original_max_len < 1:
            raise ValueError("Precondition failed: invalid scaling parameters")

        half_d = d_model // 2
        extended_len = int(scale_factor * original_max_len)

        if method == "linear_interpolation":
            scaled_freqs = [1.0 / (scale_factor * math.pow(base, (2.0 * i) / float(d_model))) for i in range(half_d)]
        elif method == "ntk_aware":
            base_new = base * math.pow(scale_factor, float(d_model) / float(d_model - 2))
            scaled_freqs = [1.0 / math.pow(base_new, (2.0 * i) / float(d_model)) for i in range(half_d)]
        elif method == "yarn":
            scaled_freqs = []
            for i in range(half_d):
                freq = 1.0 / math.pow(base, (2.0 * i) / float(d_model))
                wavelength = 2.0 * math.pi / freq
                if wavelength < original_max_len / 4.0:
                    scaled_freqs.append(freq)
                elif wavelength > original_max_len:
                    scaled_freqs.append(freq / scale_factor)
                else:
                    alpha = (original_max_len - wavelength) / (3.0 * original_max_len / 4.0)
                    interpolated = (1.0 - alpha) * (freq / scale_factor) + alpha * freq
                    scaled_freqs.append(interpolated)
        else:
            raise ValueError(f"Unknown method {method}")

        return {
            "extended_max_len": extended_len,
            "scaled_frequencies": scaled_freqs,
            "effective_scale": float(scale_factor),
        }
