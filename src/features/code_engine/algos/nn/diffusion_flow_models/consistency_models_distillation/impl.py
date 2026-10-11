from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoConsistencyModelsDistillation:
    """
    ---
    contract:
      algo_id: ALGO-NN-169
      name: NnAlgoConsistencyModelsDistillation
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - consistency_models
        - few_step_distillation
        - trajectory_invariance
        - one_step_generation
      inputs:
        type: object
        required:
          - state_xt
          - raw_model_output
          - t_step
        properties:
          state_xt:
            type: array
            items:
              type: number
            description: Current trajectory state vector x_t of dimension D.
          raw_model_output:
            type: array
            items:
              type: number
            description: Free unconstrained network output F_theta(x_t, t) of dimension D.
          t_step:
            type: number
            minimum: 0.002
            description: Continuous time coordinate t > epsilon.
          target_teacher_state:
            type: array
            items:
              type: number
            description: Teacher EMA consistency destination f_target(x_prev, t_prev) of dimension D.
          epsilon:
            type: number
            default: 0.002
            description: Smallest terminal boundary time coordinate.
          sigma_data:
            type: number
            default: 0.5
            description: Empirical data standard deviation scale.
      outputs:
        type: object
        required:
          - consistency_output
          - c_skip
          - c_out
          - consistency_loss
        properties:
          consistency_output:
            type: array
            items:
              type: number
            description: Boundary-conditioned mapping f_theta(x_t, t) predicting clean data origin.
          c_skip:
            type: number
            description: Skip scaling factor ensuring exact identity at boundary t = epsilon.
          c_out:
            type: number
            description: Output residual scaling factor vanishing at boundary t = epsilon.
          consistency_loss:
            type: number
            description: Mean squared discrepancy against teacher target (or zero if omitted).
      parameters: {}
      input_assumptions:
        - state_xt and raw_model_output have matching positive dimension D
        - t_step >= epsilon > 0
        - sigma_data > 0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact boundary parameterization"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.state_xt) > 0 and len(input.state_xt) == len(input.raw_model_output)
        - input.t_step >= input.epsilon > 0.0
        - input.sigma_data > 0.0
      postconditions:
        - len(output.consistency_output) == len(input.state_xt)
        - output.c_skip >= 0.0
        - output.consistency_loss >= 0.0
      certificate: "When t_step equals epsilon, consistency_output strictly equals state_xt"
      compatible_adapters:
        - ADAPTER-CONSISTENCY-DISTILLER
        - ADAPTER-ONE-STEP-SAMPLER
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-163
        - ALGO-NN-168
      references:
        - "https://arxiv.org/abs/2303.01469"
    ---
    """

    @staticmethod
    def forward(
        state_xt: List[float],
        raw_model_output: List[float],
        t_step: float,
        target_teacher_state: List[float] | None = None,
        epsilon: float = 0.002,
        sigma_data: float = 0.5,
    ) -> Dict[str, Any]:
        D = len(state_xt)
        if D == 0 or len(raw_model_output) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be non-empty")
        if t_step < epsilon or epsilon <= 0.0 or sigma_data <= 0.0:
            raise ValueError("Precondition failed: invalid t_step, epsilon, or sigma_data")

        # Boundary condition multipliers:
        # c_skip(t) = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2)
        # c_out(t) = (sigma_data * (t - epsilon)) / sqrt(sigma_data^2 + t^2)
        sigma_sq = sigma_data * sigma_data
        diff_t = t_step - epsilon

        c_skip = sigma_sq / (diff_t * diff_t + sigma_sq)
        c_out = (sigma_data * diff_t) / math.sqrt(sigma_sq + t_step * t_step)

        # f_theta(x_t, t) = c_skip(t) * x_t + c_out(t) * F_theta(x_t, t)
        consistency_out: List[float] = []
        for i in range(D):
            val = c_skip * state_xt[i] + c_out * raw_model_output[i]
            consistency_out.append(val)

        # Distillation / Consistency loss against target
        loss = 0.0
        if target_teacher_state is not None:
            if len(target_teacher_state) != D:
                raise ValueError("Precondition failed: target_teacher_state dimension mismatch")
            sq_sum = 0.0
            for i in range(D):
                d = consistency_out[i] - target_teacher_state[i]
                sq_sum += d * d
            loss = sq_sum / float(D)

        return {
            "consistency_output": consistency_out,
            "c_skip": c_skip,
            "c_out": c_out,
            "consistency_loss": loss,
        }
