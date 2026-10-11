from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoFastDiffusionSamplers:
    """
    ---
    contract:
      algo_id: ALGO-NN-164
      name: NnAlgoFastDiffusionSamplers
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - diffusion_sampler
        - heun_method
        - euler_integrator
        - dpm_solver
      inputs:
        type: object
        required:
          - state_xt
          - velocity_t
          - velocity_corrector_s
          - h_step
        properties:
          state_xt:
            type: array
            items:
              type: number
            description: Current state vector x_t of dimension D.
          velocity_t:
            type: array
            items:
              type: number
            description: Primary ODE velocity / drift vector v(x_t, t) of dimension D.
          velocity_corrector_s:
            type: array
            items:
              type: number
            description: Second-stage evaluated velocity v(x_tilde_s, s) of dimension D.
          h_step:
            type: number
            minimum: 0.00001
            description: Integration decrement step size h > 0.
      outputs:
        type: object
        required:
          - euler_step
          - heun_step
          - local_truncation_error
        properties:
          euler_step:
            type: array
            items:
              type: number
            description: First-order forward Euler integrated state.
          heun_step:
            type: array
            items:
              type: number
            description: Second-order predictor-corrector Heun integrated state.
          local_truncation_error:
            type: number
            description: Root mean square local truncation discrepancy between Euler and Heun.
      parameters: {}
      input_assumptions:
        - state_xt, velocity_t, and velocity_corrector_s have matching dimension D
        - h_step > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Second-order O(h^2) local truncation error"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.state_xt) > 0
        - len(input.state_xt) == len(input.velocity_t) == len(input.velocity_corrector_s)
        - input.h_step > 0.0
      postconditions:
        - len(output.euler_step) == len(input.state_xt)
        - len(output.heun_step) == len(input.state_xt)
        - output.local_truncation_error >= 0.0
      certificate: "Heun step equals state_xt - 0.5 * h_step * (velocity_t + velocity_corrector_s)"
      compatible_adapters:
        - ADAPTER-HIGH-ORDER-DIFFUSION-SOLVER
      related_algos:
        - ALGO-NN-162
        - ALGO-NN-163
        - ALGO-NN-168
      references:
        - "https://arxiv.org/abs/2206.00927"
    ---
    """

    @staticmethod
    def forward(
        state_xt: List[float],
        velocity_t: List[float],
        velocity_corrector_s: List[float],
        h_step: float,
    ) -> Dict[str, Any]:
        D = len(state_xt)
        if D == 0 or len(velocity_t) != D or len(velocity_corrector_s) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be non-empty")
        if h_step <= 0.0:
            raise ValueError("Precondition failed: h_step must be positive")

        # 1. First-order Euler step: x_{t - h} = x_t - h * v_t
        euler_step: List[float] = []
        for i in range(D):
            euler_step.append(state_xt[i] - h_step * velocity_t[i])

        # 2. Second-order Heun step: x_{t - h} = x_t - (h / 2) * (v_t + v_s)
        heun_step: List[float] = []
        error_sq_sum = 0.0
        for i in range(D):
            avg_v = 0.5 * (velocity_t[i] + velocity_corrector_s[i])
            val = state_xt[i] - h_step * avg_v
            heun_step.append(val)
            diff = val - euler_step[i]
            error_sq_sum += diff * diff

        local_error = math.sqrt(error_sq_sum / float(D))

        return {
            "euler_step": euler_step,
            "heun_step": heun_step,
            "local_truncation_error": local_error,
        }
