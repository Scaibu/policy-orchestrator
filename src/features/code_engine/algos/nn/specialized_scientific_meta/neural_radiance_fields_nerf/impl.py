from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNeuralRadianceFieldsNerf:
    """
    ---
    contract:
      algo_id: ALGO-NN-180
      name: NnAlgoNeuralRadianceFieldsNerf
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - nerf
        - neural_radiance_fields
        - volume_rendering
        - transmittance_quadrature
      inputs:
        type: object
        required:
          - sample_densities
          - sample_colors
          - sample_distances
        properties:
          sample_densities:
            type: array
            items:
              type: number
            description: Evaluated volume densities sigma_i >= 0 along ray of length N_samples.
          sample_colors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Evaluated view-dependent RGB colors c_i of shape (N_samples, 3).
          sample_distances:
            type: array
            items:
              type: number
            description: Ray step intervals delta_i = t_{i+1} - t_i of length N_samples.
          sample_t_coords:
            type: array
            items:
              type: number
            description: Optional ray metric depths t_i of length N_samples for expected depth synthesis.
      outputs:
        type: object
        required:
          - composited_rgb
          - accumulated_transmittance
          - total_opacity
          - quadrature_weights
        properties:
          composited_rgb:
            type: array
            items:
              type: number
            description: Integrated ray color vector C(r) = sum_i w_i * c_i of length 3.
          accumulated_transmittance:
            type: number
            description: Terminal transmittance T(t_f) remaining after passing through ray.
          total_opacity:
            type: number
            description: Total accumulated opacity sum_i w_i in [0, 1].
          quadrature_weights:
            type: array
            items:
              type: number
            description: Compositing alpha weights w_i = T_i * (1 - exp(-sigma_i * delta_i)).
          expected_depth:
            type: number
            description: Expected depth along ray sum_i w_i * t_i (if t_coords provided).
      parameters: {}
      input_assumptions:
        - sample_densities and sample_distances have matching length N_samples >= 1
        - sample_colors has shape (N_samples, 3)
        - densities >= 0 and distances > 0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact discrete volume quadrature"
      uses_model: false
      complexity:
        variables:
          N: number of ray sample points
        time_worst: O(N)
        time_typical: O(N)
        space: O(N)
      preconditions:
        - len(input.sample_densities) > 0
        - len(input.sample_densities) == len(input.sample_colors) == len(input.sample_distances)
        - all(len(c) == 3 for c in input.sample_colors)
      postconditions:
        - len(output.composited_rgb) == 3
        - 0.0 <= output.accumulated_transmittance <= 1.000001
        - 0.0 <= output.total_opacity <= 1.000001
      certificate: "Total opacity plus terminal transmittance strictly equals 1.0 within float precision"
      compatible_adapters:
        - ADAPTER-NERF-VOLUME-RENDERER
        - ADAPTER-INSTANT-NGP-RAYMARCHER
      related_algos:
        - ALGO-NN-181
      references:
        - "https://arxiv.org/abs/2003.08934"
    ---
    """

    @classmethod
    def forward(
        cls,
        sample_densities: List[float],
        sample_colors: List[List[float]],
        sample_distances: List[float],
        sample_t_coords: List[float] | None = None,
    ) -> Dict[str, Any]:
        N = len(sample_densities)
        if N == 0:
            raise ValueError("Precondition failed: ray samples cannot be empty")
        if len(sample_colors) != N or len(sample_distances) != N:
            raise ValueError("Precondition failed: input array lengths must match")
        if any(len(c) != 3 for c in sample_colors):
            raise ValueError("Precondition failed: sample_colors must be RGB vectors of length 3")

        # Discrete volume rendering quadrature:
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # w_i = T_i * alpha_i
        # T_{i+1} = T_i * (1 - alpha_i)
        T_i = 1.0
        weights: List[float] = []
        rgb_accum = [0.0, 0.0, 0.0]
        opacity_accum = 0.0
        expected_depth = 0.0

        for i in range(N):
            sigma = max(0.0, sample_densities[i])
            delta = max(0.0, sample_distances[i])

            # Alpha from extinction
            alpha = 1.0 - math.exp(-sigma * delta)
            w = T_i * alpha
            weights.append(w)
            opacity_accum += w

            c_rgb = sample_colors[i]
            rgb_accum[0] += w * c_rgb[0]
            rgb_accum[1] += w * c_rgb[1]
            rgb_accum[2] += w * c_rgb[2]

            if sample_t_coords is not None and i < len(sample_t_coords):
                expected_depth += w * sample_t_coords[i]

            # Update transmittance for next sample
            T_i *= (1.0 - alpha)

        res: Dict[str, Any] = {
            "composited_rgb": rgb_accum,
            "accumulated_transmittance": T_i,
            "total_opacity": opacity_accum,
            "quadrature_weights": weights,
        }
        if sample_t_coords is not None:
            res["expected_depth"] = expected_depth

        return res
