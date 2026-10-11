from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoScoreBasedSde:
    """
    ---
    contract:
      algo_id: ALGO-NN-162
      name: NnAlgoScoreBasedSde
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - score_based_generative
        - sde
        - probability_flow_ode
        - langevin_dynamics
      inputs:
        type: object
        required:
          - state_x
          - score_vector
          - beta_t
          - dt
          - brownian_noise
        properties:
          state_x:
            type: array
            items:
              type: number
            description: Current spatial state vector x_t of dimension D.
          score_vector:
            type: array
            items:
              type: number
            description: Predicted score grad_x log p_t(x_t) of dimension D.
          beta_t:
            type: number
            minimum: 0.0001
            description: VP-SDE continuous diffusion coefficient beta(t).
          dt:
            type: number
            minimum: 0.00001
            description: Time integration step size dt > 0.
          brownian_noise:
            type: array
            items:
              type: number
            description: Standard Gaussian sample for Brownian increment of dimension D.
      outputs:
        type: object
        required:
          - sde_reverse_step
          - ode_flow_drift
          - ode_reverse_step
          - score_norm
        properties:
          sde_reverse_step:
            type: array
            items:
              type: number
            description: Stochastic reverse-time SDE state updated by dt.
          ode_flow_drift:
            type: array
            items:
              type: number
            description: Instantaneous deterministic velocity vector of the Probability Flow ODE.
          ode_reverse_step:
            type: array
            items:
              type: number
            description: Deterministic Euler-integrated reverse state on the ODE trajectory.
          score_norm:
            type: number
            description: L2 Euclidean norm of the predicted score vector.
      parameters: {}
      input_assumptions:
        - state_x, score_vector, and brownian_noise have matching positive length D
        - beta_t > 0 and dt > 0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact Euler-Maruyama discretization"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.state_x) > 0
        - len(input.state_x) == len(input.score_vector) == len(input.brownian_noise)
        - input.beta_t > 0.0
        - input.dt > 0.0
      postconditions:
        - len(output.sde_reverse_step) == len(input.state_x)
        - len(output.ode_reverse_step) == len(input.state_x)
        - output.score_norm >= 0.0
      certificate: "ODE drift equals -0.5 * beta_t * (state_x + score_vector)"
      compatible_adapters:
        - ADAPTER-SDE-SOLVER
        - ADAPTER-PROBABILITY-FLOW-INTEGRATOR
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-163
        - ALGO-NN-164
      references:
        - "https://arxiv.org/abs/2011.13456"
    ---
    """

    @staticmethod
    def forward(
        state_x: List[float],
        score_vector: List[float],
        beta_t: float,
        dt: float,
        brownian_noise: List[float],
    ) -> Dict[str, Any]:
        D = len(state_x)
        if D == 0 or len(score_vector) != D or len(brownian_noise) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be non-empty")
        if beta_t <= 0.0 or dt <= 0.0:
            raise ValueError("Precondition failed: beta_t and dt must be strictly positive")

        # In reverse time from t to t - dt:
        # VP-SDE Drift: f(x, t) - g(t)^2 * score = -0.5 * beta_t * x - beta_t * score
        # Probability Flow ODE Drift: f(x, t) - 0.5 * g(t)^2 * score = -0.5 * beta_t * x - 0.5 * beta_t * score
        # Euler integration backward in time (dt subtraction):
        # x_{t - dt} = x_t - drift * dt + g(t) * sqrt(dt) * noise (for SDE)

        g_t = math.sqrt(beta_t)
        sqrt_dt = math.sqrt(dt)

        sde_step: List[float] = []
        ode_drift: List[float] = []
        ode_step: List[float] = []
        score_sq_sum = 0.0

        for i in range(D):
            x = state_x[i]
            s = score_vector[i]
            dw = brownian_noise[i]

            score_sq_sum += s * s

            # SDE drift:
            f_sde = -0.5 * beta_t * x - beta_t * s
            # ODE drift:
            f_ode = -0.5 * beta_t * x - 0.5 * beta_t * s
            ode_drift.append(f_ode)

            # Reverse step (subtracting drift dt because stepping backward from t to t - dt)
            # In standard reverse-time SDE notation: dx = [f - g^2 score] dt + g dW
            # Stepping backward: x_{t - dt} = x_t - [f - g^2 score] * dt + g * sqrt(dt) * noise
            next_x_sde = x - f_sde * dt + g_t * sqrt_dt * dw
            sde_step.append(next_x_sde)

            next_x_ode = x - f_ode * dt
            ode_step.append(next_x_ode)

        score_norm = math.sqrt(score_sq_sum)

        return {
            "sde_reverse_step": sde_step,
            "ode_flow_drift": ode_drift,
            "ode_reverse_step": ode_step,
            "score_norm": score_norm,
        }
