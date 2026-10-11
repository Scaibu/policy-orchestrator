from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoFlowMatchingRectifiedFlow:
    """
    ---
    contract:
      algo_id: ALGO-NN-168
      name: NnAlgoFlowMatchingRectifiedFlow
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - flow_matching
        - rectified_flow
        - continuous_normalizing_flow
        - straight_line_ode
      inputs:
        type: object
        required:
          - noise_sample_x0
          - data_sample_x1
          - predicted_velocity
          - timestep_t
        properties:
          noise_sample_x0:
            type: array
            items:
              type: number
            description: Base distribution sample x_0 (noise) of dimension D.
          data_sample_x1:
            type: array
            items:
              type: number
            description: Target data manifold sample x_1 of dimension D.
          predicted_velocity:
            type: array
            items:
              type: number
            description: Model predicted vector field v_theta(x_t, t) of dimension D.
          timestep_t:
            type: number
            minimum: 0.0
            maximum: 1.0
            description: Flow time coordinate t in [0.0, 1.0].
      outputs:
        type: object
        required:
          - interpolated_xt
          - target_velocity
          - velocity_matching_loss
          - straight_step_x1
        properties:
          interpolated_xt:
            type: array
            items:
              type: number
            description: Straight-line interpolated state x_t = (1 - t) * x_0 + t * x_1.
          target_velocity:
            type: array
            items:
              type: number
            description: Exact theoretical path velocity u_t = x_1 - x_0.
          velocity_matching_loss:
            type: number
            description: Mean squared flow matching objective ||v_theta - u_t||^2 / D.
          straight_step_x1:
            type: array
            items:
              type: number
            description: One-step straight line projected sample x_t + (1 - t) * v_theta.
      parameters: {}
      input_assumptions:
        - noise_sample_x0, data_sample_x1, and predicted_velocity have matching dimension D
        - 0.0 <= timestep_t <= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact within IEEE-754 precision"
      uses_model: false
      complexity:
        variables:
          D: state vector dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.noise_sample_x0) > 0
        - len(input.noise_sample_x0) == len(input.data_sample_x1) == len(input.predicted_velocity)
        - 0.0 <= input.timestep_t <= 1.0
      postconditions:
        - len(output.interpolated_xt) == len(input.noise_sample_x0)
        - len(output.target_velocity) == len(input.noise_sample_x0)
        - output.velocity_matching_loss >= 0.0
      certificate: "Target velocity strictly equals data_sample_x1 - noise_sample_x0"
      compatible_adapters:
        - ADAPTER-FLOW-MATCHING-TRAINER
        - ADAPTER-RECTIFIED-FLOW-SOLVER
      related_algos:
        - ALGO-NN-158
        - ALGO-NN-161
        - ALGO-NN-164
      references:
        - "https://arxiv.org/abs/2210.02747"
        - "https://arxiv.org/abs/2209.03003"
    ---
    """

    @staticmethod
    def forward(
        noise_sample_x0: List[float],
        data_sample_x1: List[float],
        predicted_velocity: List[float],
        timestep_t: float,
    ) -> Dict[str, Any]:
        D = len(noise_sample_x0)
        if D == 0 or len(data_sample_x1) != D or len(predicted_velocity) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be positive")
        if not (0.0 <= timestep_t <= 1.0):
            raise ValueError("Precondition failed: timestep_t must be in [0, 1]")

        t = timestep_t
        one_minus_t = 1.0 - t

        interpolated_xt: List[float] = []
        target_velocity: List[float] = []
        straight_step_x1: List[float] = []
        loss_sq_sum = 0.0

        for i in range(D):
            x0 = noise_sample_x0[i]
            x1 = data_sample_x1[i]
            v_pred = predicted_velocity[i]

            # Straight path: x_t = (1 - t) * x_0 + t * x_1
            xt = one_minus_t * x0 + t * x1
            interpolated_xt.append(xt)

            # Target velocity: dx_t / dt = x_1 - x_0
            u_target = x1 - x0
            target_velocity.append(u_target)

            diff = v_pred - u_target
            loss_sq_sum += diff * diff

            # One-step straight projection: x_1_hat = x_t + (1 - t) * v_pred
            straight_step_x1.append(xt + one_minus_t * v_pred)

        loss = loss_sq_sum / float(D)

        return {
            "interpolated_xt": interpolated_xt,
            "target_velocity": target_velocity,
            "velocity_matching_loss": loss,
            "straight_step_x1": straight_step_x1,
        }
