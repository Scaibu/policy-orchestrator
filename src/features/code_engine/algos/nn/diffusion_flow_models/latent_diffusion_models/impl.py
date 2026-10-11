from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLatentDiffusionModels:
    """
    ---
    contract:
      algo_id: ALGO-NN-166
      name: NnAlgoLatentDiffusionModels
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - latent_diffusion
        - stable_diffusion
        - spatial_compression
        - cross_attention_conditioning
      inputs:
        type: object
        required:
          - latent_z0
          - latent_noise
          - alpha_bar_t
        properties:
          latent_z0:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoded latent representation z_0 of shape (N_lat, d_lat).
          latent_noise:
            type: array
            items:
              type: array
              items:
                type: number
            description: Latent Gaussian noise tensor of shape (N_lat, d_lat).
          alpha_bar_t:
            type: number
            minimum: 0.0001
            maximum: 0.9999
            description: Cumulative noise scale alpha_bar at timestep t.
          compression_factor:
            type: integer
            default: 8
            description: Spatial compression downsampling ratio f = H / h.
          scale_factor:
            type: number
            default: 0.18215
            description: Latent variance standardization multiplier.
      outputs:
        type: object
        required:
          - diffused_latent_zt
          - scaled_latent_z0
          - memory_reduction_ratio
          - latent_energy
        properties:
          diffused_latent_zt:
            type: array
            items:
              type: array
              items:
                type: number
            description: Diffused noisy latent tensor z_t at timestep t.
          scaled_latent_z0:
            type: array
            items:
              type: array
              items:
                type: number
            description: Variance-standardized latent representation.
          memory_reduction_ratio:
            type: number
            description: Theoretical spatial memory savings factor f^2.
          latent_energy:
            type: number
            description: Total Frobenius energy norm of diffused latent.
      parameters: {}
      input_assumptions:
        - latent_z0 and latent_noise are non-empty with matching uniform dimensions
        - 0.0 < alpha_bar_t < 1.0
        - compression_factor >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact floating-point operations"
      uses_model: false
      complexity:
        variables:
          N: latent tokens / spatial points
          D: latent channel dimension
        time_worst: O(N * D)
        time_typical: O(N * D)
        space: O(N * D)
      preconditions:
        - len(input.latent_z0) > 0 and len(input.latent_z0[0]) > 0
        - len(input.latent_z0) == len(input.latent_noise) and len(input.latent_z0[0]) == len(input.latent_noise[0])
        - 0.0 < input.alpha_bar_t < 1.0
        - input.compression_factor >= 1
      postconditions:
        - len(output.diffused_latent_zt) == len(input.latent_z0)
        - len(output.diffused_latent_zt[0]) == len(input.latent_z0[0])
        - output.memory_reduction_ratio >= 1.0
      certificate: "Diffused latent follows z_t = sqrt(alpha_bar) * z_scaled + sqrt(1 - alpha_bar) * noise"
      compatible_adapters:
        - ADAPTER-LATENT-DIFFUSION-PIPELINE
      related_algos:
        - ALGO-NN-152
        - ALGO-NN-161
        - ALGO-NN-167
      references:
        - "https://arxiv.org/abs/2112.10752"
    ---
    """

    @staticmethod
    def forward(
        latent_z0: List[List[float]],
        latent_noise: List[List[float]],
        alpha_bar_t: float,
        compression_factor: int = 8,
        scale_factor: float = 0.18215,
    ) -> Dict[str, Any]:
        N_lat = len(latent_z0)
        if N_lat == 0 or len(latent_z0[0]) == 0:
            raise ValueError("Precondition failed: latent_z0 cannot be empty")
        d_lat = len(latent_z0[0])

        if len(latent_noise) != N_lat or any(len(r) != d_lat for r in latent_noise):
            raise ValueError("Precondition failed: latent_noise dimension mismatch")
        if not (0.0 < alpha_bar_t < 1.0):
            raise ValueError("Precondition failed: alpha_bar_t must be in (0, 1)")
        if compression_factor < 1:
            raise ValueError("Precondition failed: compression_factor must be >= 1")

        sqrt_ab = math.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = math.sqrt(1.0 - alpha_bar_t)

        scaled_z0: List[List[float]] = []
        diffused_zt: List[List[float]] = []
        energy_sum = 0.0

        for i in range(N_lat):
            row_scaled: List[float] = []
            row_zt: List[float] = []
            for j in range(d_lat):
                s_val = latent_z0[i][j] * scale_factor
                row_scaled.append(s_val)

                z_t_val = sqrt_ab * s_val + sqrt_one_minus_ab * latent_noise[i][j]
                row_zt.append(z_t_val)
                energy_sum += z_t_val * z_t_val

            scaled_z0.append(row_scaled)
            diffused_zt.append(row_zt)

        reduction_ratio = float(compression_factor * compression_factor)

        return {
            "diffused_latent_zt": diffused_zt,
            "scaled_latent_z0": scaled_z0,
            "memory_reduction_ratio": reduction_ratio,
            "latent_energy": energy_sum,
        }
