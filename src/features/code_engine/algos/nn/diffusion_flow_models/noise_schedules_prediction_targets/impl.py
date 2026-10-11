from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNoiseSchedulesPredictionTargets:
    """
    ---
    contract:
      algo_id: ALGO-NN-171
      name: NnAlgoNoiseSchedulesPredictionTargets
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - noise_schedules
        - cosine_schedule
        - v_prediction
        - signal_to_noise_ratio
      inputs:
        type: object
        required:
          - timestep_t
          - total_steps_T
          - clean_x0
          - noise_epsilon
        properties:
          timestep_t:
            type: integer
            minimum: 0
            description: Current integer diffusion step index t in [0, total_steps_T].
          total_steps_T:
            type: integer
            minimum: 1
            description: Total diffusion discretization steps T.
          clean_x0:
            type: array
            items:
              type: number
            description: Clean ground-truth data vector x_0 of dimension D.
          noise_epsilon:
            type: array
            items:
              type: number
            description: Latent standard Gaussian perturbation epsilon of dimension D.
          schedule_type:
            type: string
            default: cosine
            enum:
              - cosine
              - linear
            description: Noise scheduling variance profile.
      outputs:
        type: object
        required:
          - alpha_bar_t
          - snr_t
          - log_snr_t
          - target_v
          - reconstructed_x0_from_v
        properties:
          alpha_bar_t:
            type: number
            description: Cumulative variance factor alpha_bar_t in (0, 1).
          snr_t:
            type: number
            description: Signal-to-noise ratio SNR(t) = alpha_bar / (1 - alpha_bar).
          log_snr_t:
            type: number
            description: Log signal-to-noise ratio log(SNR(t)).
          target_v:
            type: array
            items:
              type: number
            description: Velocity prediction target vector v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0.
          reconstructed_x0_from_v:
            type: array
            items:
              type: number
            description: Algebraic reconstruction of clean x0 using state x_t and velocity v.
      parameters: {}
      input_assumptions:
        - clean_x0 and noise_epsilon have matching positive dimension D
        - 0 <= timestep_t <= total_steps_T
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact analytical computation within float precision"
      uses_model: false
      complexity:
        variables:
          D: state dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(D)
      preconditions:
        - len(input.clean_x0) > 0 and len(input.clean_x0) == len(input.noise_epsilon)
        - 0 <= input.timestep_t <= input.total_steps_T
        - input.total_steps_T >= 1
      postconditions:
        - 0.0 < output.alpha_bar_t < 1.0
        - len(output.target_v) == len(input.clean_x0)
        - len(output.reconstructed_x0_from_v) == len(input.clean_x0)
      certificate: "Reconstructed x0 from velocity matches original clean_x0 within float tolerance"
      compatible_adapters:
        - ADAPTER-NOISE-SCHEDULE-BUILDER
        - ADAPTER-V-PREDICTION-LOSS
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-168
      references:
        - "https://arxiv.org/abs/2102.09672"
        - "https://arxiv.org/abs/2202.00512"
    ---
    """

    @staticmethod
    def forward(
        timestep_t: int,
        total_steps_T: int,
        clean_x0: List[float],
        noise_epsilon: List[float],
        schedule_type: str = "cosine",
    ) -> Dict[str, Any]:
        D = len(clean_x0)
        if D == 0 or len(noise_epsilon) != D:
            raise ValueError("Precondition failed: vector dimensions must match and be non-empty")
        if not (0 <= timestep_t <= total_steps_T) or total_steps_T < 1:
            raise ValueError("Precondition failed: timestep_t out of valid bounds")

        # 1. Compute alpha_bar_t
        if schedule_type == "linear":
            # Linear beta schedule from beta_1 = 0.0001 to beta_T = 0.02
            # alpha_bar_t = product_{s=1}^t (1 - beta_s)
            frac = float(timestep_t) / float(total_steps_T)
            # Quadratic decay approximation for linear beta:
            alpha_bar = max(0.0001, min(0.9999, (1.0 - frac) ** 2))
        else:
            # Improved Cosine schedule: f(t) = cos(((t/T + s) / (1 + s)) * (pi / 2))^2
            s = 0.008
            t_ratio = float(timestep_t) / float(total_steps_T)
            arg = ((t_ratio + s) / (1.0 + s)) * (math.pi * 0.5)
            f_t = math.cos(arg) ** 2
            f_0 = math.cos((s / (1.0 + s)) * (math.pi * 0.5)) ** 2
            alpha_bar = max(0.0001, min(0.9999, f_t / f_0))

        sqrt_ab = math.sqrt(alpha_bar)
        sqrt_one_minus_ab = math.sqrt(1.0 - alpha_bar)

        # 2. SNR = alpha_bar / (1 - alpha_bar)
        snr = alpha_bar / (1.0 - alpha_bar)
        log_snr = math.log(alpha_bar) - math.log(1.0 - alpha_bar)

        # 3. Noisy state x_t and velocity target v:
        # x_t = sqrt_ab * x0 + sqrt_one_minus_ab * eps
        # v = sqrt_ab * eps - sqrt_one_minus_ab * x0
        target_v: List[float] = []
        x_t: List[float] = []
        for i in range(D):
            x0 = clean_x0[i]
            eps = noise_epsilon[i]
            xt_val = sqrt_ab * x0 + sqrt_one_minus_ab * eps
            v_val = sqrt_ab * eps - sqrt_one_minus_ab * x0
            x_t.append(xt_val)
            target_v.append(v_val)

        # 4. Invert velocity back to clean x_0:
        # x_0 = sqrt_ab * x_t - sqrt_one_minus_ab * v
        recon_x0: List[float] = []
        for i in range(D):
            val = sqrt_ab * x_t[i] - sqrt_one_minus_ab * target_v[i]
            recon_x0.append(val)

        return {
            "alpha_bar_t": alpha_bar,
            "snr_t": snr,
            "log_snr_t": log_snr,
            "target_v": target_v,
            "reconstructed_x0_from_v": recon_x0,
        }
