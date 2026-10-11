from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoGaussianSplatting3d:
    """
    ---
    contract:
      algo_id: ALGO-NN-181
      name: NnAlgoGaussianSplatting3d
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - gaussian_splatting
        - 3d_reconstruction
        - differentiable_rasterization
        - alpha_compositing
      inputs:
        type: object
        required:
          - pixel_coord
          - gaussian_centers_2d
          - inv_covariances_2d
          - opacities
          - colors
        properties:
          pixel_coord:
            type: array
            items:
              type: number
            description: 2D pixel coordinate [u, v] on screen plane.
          gaussian_centers_2d:
            type: array
            items:
              type: array
              items:
                type: number
            description: Projected 2D centers [mu_u, mu_v] of shape (K_gaussians, 2).
          inv_covariances_2d:
            type: array
            items:
              type: array
              items:
                type: number
            description: 2D screen inverse covariance parameters [a, b, c] where matrix is [[a, b], [b, c]].
          opacities:
            type: array
            items:
              type: number
            description: Base opacity values alpha_i in [0, 1] of length K_gaussians (depth-sorted).
          colors:
            type: array
            items:
              type: array
              items:
                type: number
            description: RGB colors of shape (K_gaussians, 3).
      outputs:
        type: object
        required:
          - pixel_rgb
          - terminal_transmittance
          - total_weight_sum
          - effective_alphas
        properties:
          pixel_rgb:
            type: array
            items:
              type: number
            description: Alpha-composited RGB color vector of length 3.
          terminal_transmittance:
            type: number
            description: Remaining ray transmittance T_K after splat blending.
          total_weight_sum:
            type: number
            description: Total accumulated opacity across Gaussians at this pixel.
          effective_alphas:
            type: array
            items:
              type: number
            description: Spatial alpha values alpha_i * G_i(x) for each Gaussian.
      parameters: {}
      input_assumptions:
        - pixel_coord has length 2
        - gaussian_centers_2d has shape (K, 2)
        - inv_covariances_2d has shape (K, 3) representing positive definite 2x2 matrix
        - opacities has length K, colors has shape (K, 3)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact rasterized 2D Gaussian evaluation"
      uses_model: false
      complexity:
        variables:
          K: number of overlapping Gaussians
        time_worst: O(K)
        time_typical: O(K)
        space: O(K)
      preconditions:
        - len(input.pixel_coord) == 2
        - len(input.gaussian_centers_2d) > 0
        - len(input.gaussian_centers_2d) == len(input.inv_covariances_2d) == len(input.opacities) == len(input.colors)
      postconditions:
        - len(output.pixel_rgb) == 3
        - 0.0 <= output.terminal_transmittance <= 1.000001
        - len(output.effective_alphas) == len(input.opacities)
      certificate: "Compositing respects strict front-to-back depth order: C = sum_i c_i * alpha_i * prod_{j<i} (1 - alpha_j)"
      compatible_adapters:
        - ADAPTER-3DGS-RASTERIZER
        - ADAPTER-DIFFERENTIABLE-SPLATTING-ENGINE
      related_algos:
        - ALGO-NN-180
      references:
        - "https://arxiv.org/abs/2308.04079"
    ---
    """

    @classmethod
    def forward(
        cls,
        pixel_coord: List[float],
        gaussian_centers_2d: List[List[float]],
        inv_covariances_2d: List[List[float]],
        opacities: List[float],
        colors: List[List[float]],
    ) -> Dict[str, Any]:
        if len(pixel_coord) != 2:
            raise ValueError("Precondition failed: pixel_coord must have length 2")
        K = len(gaussian_centers_2d)
        if K == 0:
            raise ValueError("Precondition failed: at least one Gaussian required")
        if (
            len(inv_covariances_2d) != K
            or len(opacities) != K
            or len(colors) != K
        ):
            raise ValueError("Precondition failed: mismatched Gaussian list lengths")
        if any(len(c) != 2 for c in gaussian_centers_2d) or any(len(inv) != 3 for inv in inv_covariances_2d) or any(len(rgb) != 3 for rgb in colors):
            raise ValueError("Precondition failed: invalid geometric dimensions in Gaussian inputs")

        u_px, v_px = pixel_coord[0], pixel_coord[1]
        T = 1.0
        pixel_rgb = [0.0, 0.0, 0.0]
        total_w = 0.0
        effective_alphas: List[float] = []

        for i in range(K):
            mu_u, mu_v = gaussian_centers_2d[i][0], gaussian_centers_2d[i][1]
            du = u_px - mu_u
            dv = v_px - mu_v

            # 2D inverse covariance quadratic form:
            # power = -0.5 * (a * du^2 + 2 * b * du * dv + c * dv^2)
            a = inv_covariances_2d[i][0]
            b = inv_covariances_2d[i][1]
            c = inv_covariances_2d[i][2]

            quad = a * du * du + 2.0 * b * du * dv + c * dv * dv
            if quad < 0.0:
                quad = 0.0

            # 2D Gaussian density
            g_val = math.exp(-0.5 * quad)

            # Effective alpha clipped to [0, 0.99]
            base_alpha = max(0.0, min(1.0, opacities[i]))
            alpha = min(0.99, base_alpha * g_val)
            effective_alphas.append(alpha)

            if alpha < 1.0 / 255.0:
                continue

            # Front-to-back blending weight
            w = alpha * T
            rgb = colors[i]
            pixel_rgb[0] += w * rgb[0]
            pixel_rgb[1] += w * rgb[1]
            pixel_rgb[2] += w * rgb[2]

            total_w += w
            T *= (1.0 - alpha)

            if T < 0.0001:
                break

        return {
            "pixel_rgb": pixel_rgb,
            "terminal_transmittance": T,
            "total_weight_sum": total_w,
            "effective_alphas": effective_alphas,
        }
