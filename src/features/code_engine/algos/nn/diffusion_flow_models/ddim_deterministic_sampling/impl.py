from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDdimDeterministicSampling:
    """
    ---
    contract:
      algo_id: ALGO-NN-163
      name: NnAlgoDdimDeterministicSampling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - ddim
        - deterministic_sampling
        - non_markovian
        - latent_inversion
      inputs:
        type: object
        required:
          - state_xt
          - predicted_noise
          - alpha_bar_t
          - alpha_bar_prev
        properties:
          state_xt:
            type: array
            items:
              type: number
            description: Current noisy state vector x_t of dimension D.
          predicted_noise:
            type: array
            items:
              type: number
            description: Model predicted noise epsilon_theta of dimension D.
          alpha_bar_t:
            type: number
            minimum: 0.0001
            maximum: 0.9999
            description: Cumulative variance alpha_bar at current timestep t.
          alpha_bar_prev:
            type: number
            minimum: 0.0001
            maximum: 0.9999
            description: Cumulative variance alpha_bar at destination timestep s < t.
          eta:
            type: number
            default: 0.0
            description: Stochasticity parameter (0.0 = deterministic DDIM, 1.0 = DDPM).
      outputs:
        type: object
        required:
          - next_state_xs
          - predicted_x0
          - directional_pointing_term
          - sigma_t
        properties:
          next_state_xs:
            type: array
            items:
              type: number
            description: Sampled or integrated state x_s at earlier timestep s.
          predicted_x0:
            type: array
            items:
              type: number
            description: Reconstructed clean data estimate x_hat_0.
          directional_pointing_term:
            type: array
            items:
              type: number
            description: Tangent directional vector pointing toward x_t.
          sigma_t:
            type: number
            description: Effective stochastic variance scale.
      parameters: {}
      input_assumptions:
        - state_xt and predicted_noise have matching positive length D
        - 0.0 < alpha_bar_t < alpha_bar_prev < 1.0 (since s < t means less noise, so alpha_bar_prev > alpha_bar_t)
        - eta >= 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact non-Markovian update within float precision"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.state_xt) > 0 and len(input.state_xt) == len(input.predicted_noise)
        - 0.0 < input.alpha_bar_t < 1.0
        - 0.0 < input.alpha_bar_prev < 1.0
        - input.eta >= 0.0
      postconditions:
        - len(output.next_state_xs) == len(input.state_xt)
        - len(output.predicted_x0) == len(input.state_xt)
        - output.sigma_t >= 0.0
      certificate: "When eta is zero, next_state_xs is uniquely deterministic"
      compatible_adapters:
        - ADAPTER-DDIM-FAST-SAMPLER
        - ADAPTER-INVERSION-EDITOR
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-164
      references:
        - "https://arxiv.org/abs/2010.02502"
    ---
    """

    @staticmethod
    def forward(
        state_xt: List[float],
        predicted_noise: List[float],
        alpha_bar_t: float,
        alpha_bar_prev: float,
        eta: float = 0.0,
    ) -> Dict[str, Any]:
        D = len(state_xt)
        if D == 0 or len(predicted_noise) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be non-empty")

        if not (0.0 < alpha_bar_t < 1.0) or not (0.0 < alpha_bar_prev < 1.0):
            raise ValueError("Precondition failed: alpha_bar values must be in (0, 1)")
        if eta < 0.0:
            raise ValueError("Precondition failed: eta must be non-negative")

        sqrt_ab_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_ab_t = math.sqrt(1.0 - alpha_bar_t)

        sqrt_ab_prev = math.sqrt(alpha_bar_prev)
        one_minus_ab_prev = 1.0 - alpha_bar_prev

        # 1. Predict clean x_0: x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_x0: List[float] = []
        for i in range(D):
            val = (state_xt[i] - sqrt_one_minus_ab_t * predicted_noise[i]) / sqrt_ab_t
            pred_x0.append(val)

        # 2. Compute sigma_t:
        # sigma_t = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_prev)
        if alpha_bar_prev > alpha_bar_t and eta > 0.0:
            ratio1 = one_minus_ab_prev / (1.0 - alpha_bar_t)
            ratio2 = 1.0 - (alpha_bar_t / alpha_bar_prev)
            sigma_t = eta * math.sqrt(max(0.0, ratio1 * ratio2))
        else:
            sigma_t = 0.0

        # 3. Direction pointing to x_t: sqrt(1 - alpha_bar_prev - sigma_t^2) * eps
        c_dir = math.sqrt(max(0.0, one_minus_ab_prev - sigma_t * sigma_t))
        dir_pointing: List[float] = []
        for i in range(D):
            dir_pointing.append(c_dir * predicted_noise[i])

        # 4. Next step x_s (deterministic when eta = 0):
        # x_s = sqrt(alpha_bar_prev) * x_hat_0 + direction
        next_xs: List[float] = []
        for i in range(D):
            val = sqrt_ab_prev * pred_x0[i] + dir_pointing[i]
            next_xs.append(val)

        return {
            "next_state_xs": next_xs,
            "predicted_x0": pred_x0,
            "directional_pointing_term": dir_pointing,
            "sigma_t": sigma_t,
        }
