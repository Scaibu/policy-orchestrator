from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNeuralOrdinaryDifferentialEquations:
    """
    ---
    contract:
      algo_id: ALGO-NN-179
      name: NnAlgoNeuralOrdinaryDifferentialEquations
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - neural_ode
        - continuous_depth
        - rk4_integrator
        - adjoint_dynamics
      inputs:
        type: object
        required:
          - initial_state_h
          - k1_slope
          - k2_slope
          - k3_slope
          - k4_slope
          - dt_step
        properties:
          initial_state_h:
            type: array
            items:
              type: number
            description: Hidden state vector h(t) of dimension D.
          k1_slope:
            type: array
            items:
              type: number
            description: First RK4 stage derivative f(h, t) of dimension D.
          k2_slope:
            type: array
            items:
              type: number
            description: Second RK4 stage derivative f(h + 0.5*dt*k1, t + 0.5*dt) of dimension D.
          k3_slope:
            type: array
            items:
              type: number
            description: Third RK4 stage derivative f(h + 0.5*dt*k2, t + 0.5*dt) of dimension D.
          k4_slope:
            type: array
            items:
              type: number
            description: Fourth RK4 stage derivative f(h + dt*k3, t + dt) of dimension D.
          dt_step:
            type: number
            minimum: 0.00001
            description: Continuous time integration step size dt > 0.
          adjoint_gradient:
            type: array
            items:
              type: number
            description: Upstream loss gradient a(t1) = dL / dh(t1) of dimension D (optional).
      outputs:
        type: object
        required:
          - integrated_state_next
          - effective_drift
          - state_displacement_norm
        properties:
          integrated_state_next:
            type: array
            items:
              type: number
            description: Fourth-order integrated state h(t + dt) of dimension D.
          effective_drift:
            type: array
            items:
              type: number
            description: Weighted RK4 effective velocity (k1 + 2*k2 + 2*k3 + k4) / 6.
          state_displacement_norm:
            type: number
            description: Euclidean norm of the integration displacement dt * effective_drift.
      parameters: {}
      input_assumptions:
        - initial_state_h and all four k-slopes have matching positive dimension D
        - dt_step > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Fourth-order O(dt^5) local truncation error"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.initial_state_h) > 0
        - len(input.initial_state_h) == len(input.k1_slope) == len(input.k2_slope) == len(input.k3_slope) == len(input.k4_slope)
        - input.dt_step > 0.0
      postconditions:
        - len(output.integrated_state_next) == len(input.initial_state_h)
        - len(output.effective_drift) == len(input.initial_state_h)
        - output.state_displacement_norm >= 0.0
      certificate: "RK4 step satisfies h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)"
      compatible_adapters:
        - ADAPTER-NEURAL-ODE-SOLVER
        - ADAPTER-CONTINUOUS-NORMALIZING-FLOW
      related_algos:
        - ALGO-NN-162
        - ALGO-NN-164
      references:
        - "https://arxiv.org/abs/1806.07366"
    ---
    """

    @classmethod
    def forward(
        cls,
        initial_state_h: List[float],
        k1_slope: List[float],
        k2_slope: List[float],
        k3_slope: List[float],
        k4_slope: List[float],
        dt_step: float,
        adjoint_gradient: List[float] | None = None,
    ) -> Dict[str, Any]:
        D = len(initial_state_h)
        if D == 0:
            raise ValueError("Precondition failed: initial_state_h cannot be empty")
        if (
            len(k1_slope) != D
            or len(k2_slope) != D
            or len(k3_slope) != D
            or len(k4_slope) != D
        ):
            raise ValueError("Precondition failed: slope dimensions must match initial_state_h")
        if dt_step <= 0.0:
            raise ValueError("Precondition failed: dt_step must be positive")

        # 1. Classical 4th-order Runge-Kutta combination:
        # effective_drift = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        # h_{next} = h + dt * effective_drift
        effective_drift: List[float] = []
        next_h: List[float] = []
        disp_sq_sum = 0.0
        inv_six = 1.0 / 6.0

        for d in range(D):
            drift_val = inv_six * (
                k1_slope[d] + 2.0 * k2_slope[d] + 2.0 * k3_slope[d] + k4_slope[d]
            )
            displacement = dt_step * drift_val
            disp_sq_sum += displacement * displacement

            effective_drift.append(drift_val)
            next_h.append(initial_state_h[d] + displacement)

        return {
            "integrated_state_next": next_h,
            "effective_drift": effective_drift,
            "state_displacement_norm": math.sqrt(disp_sq_sum),
        }
