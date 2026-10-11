from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoClassifierFreeGuidance:
    """
    ---
    contract:
      algo_id: ALGO-NN-165
      name: NnAlgoClassifierFreeGuidance
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - classifier_free_guidance
        - cfg
        - prompt_adherence
        - negative_prompt
      inputs:
        type: object
        required:
          - unconditional_noise
          - conditional_noise
          - guidance_scale
        properties:
          unconditional_noise:
            type: array
            items:
              type: number
            description: Unconditional score or noise prediction epsilon(x_t, null) of dimension D.
          conditional_noise:
            type: array
            items:
              type: number
            description: Conditioned score or noise prediction epsilon(x_t, c) of dimension D.
          guidance_scale:
            type: number
            minimum: 1.0
            description: Extrapolation guidance weight w >= 1.0.
          negative_noise:
            type: array
            items:
              type: number
            description: Optional negative prompt prediction epsilon(x_t, neg) of dimension D.
      outputs:
        type: object
        required:
          - guided_noise
          - guidance_delta_norm
          - amplification_ratio
        properties:
          guided_noise:
            type: array
            items:
              type: number
            description: Extrapolated guided noise prediction vector of dimension D.
          guidance_delta_norm:
            type: number
            description: Euclidean norm of the guidance steering vector ||epsilon_cond - epsilon_uncond||.
          amplification_ratio:
            type: number
            description: Ratio of guided norm over base conditional norm.
      parameters: {}
      input_assumptions:
        - unconditional_noise and conditional_noise have matching positive length D
        - guidance_scale >= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact vector extrapolation"
      uses_model: false
      complexity:
        variables:
          D: prediction dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.unconditional_noise) > 0 and len(input.unconditional_noise) == len(input.conditional_noise)
        - input.guidance_scale >= 1.0
      postconditions:
        - len(output.guided_noise) == len(input.unconditional_noise)
        - output.guidance_delta_norm >= 0.0
      certificate: "Guided noise equals baseline + guidance_scale * (conditional - baseline)"
      compatible_adapters:
        - ADAPTER-CFG-STEERER
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-163
        - ALGO-NN-166
      references:
        - "https://arxiv.org/abs/2207.12598"
    ---
    """

    @staticmethod
    def forward(
        unconditional_noise: List[float],
        conditional_noise: List[float],
        guidance_scale: float,
        negative_noise: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        D = len(unconditional_noise)
        if D == 0 or len(conditional_noise) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be positive")
        if guidance_scale < 1.0:
            raise ValueError("Precondition failed: guidance_scale must be at least 1.0")

        base_uncond = negative_noise if negative_noise is not None else unconditional_noise
        if len(base_uncond) != D:
            raise ValueError("Precondition failed: negative_noise dimension mismatch")

        # Guided noise: eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        guided_noise: List[float] = []
        delta_sq_sum = 0.0
        cond_sq_sum = 0.0
        guided_sq_sum = 0.0

        for i in range(D):
            u = base_uncond[i]
            c = conditional_noise[i]
            delta = c - u
            delta_sq_sum += delta * delta
            cond_sq_sum += c * c

            val = u + guidance_scale * delta
            guided_noise.append(val)
            guided_sq_sum += val * val

        delta_norm = math.sqrt(delta_sq_sum)
        cond_norm = math.sqrt(cond_sq_sum)
        guided_norm = math.sqrt(guided_sq_sum)
        amp_ratio = (guided_norm / cond_norm) if cond_norm > 1e-12 else 1.0

        return {
            "guided_noise": guided_noise,
            "guidance_delta_norm": delta_norm,
            "amplification_ratio": amp_ratio,
        }
