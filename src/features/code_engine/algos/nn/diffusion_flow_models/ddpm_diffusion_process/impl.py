from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDdpmDiffusionProcess:
    """
    ---
    contract:
      algo_id: ALGO-NN-161
      name: NnAlgoDdpmDiffusionProcess
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - diffusion
        - ddpm
        - score_matching
        - generative_process
      inputs:
        type: object
        required:
          - clean_sample_x0
          - noise_epsilon
          - predicted_noise
          - alpha_bar_t
          - beta_t
        properties:
          clean_sample_x0:
            type: array
            items:
              type: number
            description: Clean data vector x_0 of dimension D.
          noise_epsilon:
            type: array
            items:
              type: number
            description: Standard Gaussian sample epsilon of dimension D.
          predicted_noise:
            type: array
            items:
              type: number
            description: Neural network predicted noise epsilon_theta of dimension D.
          alpha_bar_t:
            type: number
            minimum: 0.0001
            maximum: 0.9999
            description: Cumulative noise product alpha_bar_t at timestep t.
          beta_t:
            type: number
            minimum: 0.00001
            maximum: 0.9999
            description: Instantaneous forward variance step beta_t at timestep t.
      outputs:
        type: object
        required:
          - noisy_sample_xt
          - predicted_x0
          - noise_mse_loss
          - posterior_mean
        properties:
          noisy_sample_xt:
            type: array
            items:
              type: number
            description: Forward diffused noisy observation x_t at timestep t.
          predicted_x0:
            type: array
            items:
              type: number
            description: Tweedie-derived clean data estimate x_hat_0.
          noise_mse_loss:
            type: number
            description: Mean squared error between true and predicted noise.
          posterior_mean:
            type: array
            items:
              type: number
            description: Reverse transition Markov posterior mean mu_tilde_t.
      parameters: {}
      input_assumptions:
        - clean_sample_x0, noise_epsilon, and predicted_noise have matching positive length D
        - 0.0 < alpha_bar_t < 1.0 and 0.0 < beta_t < 1.0
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
        - len(input.clean_sample_x0) > 0
        - len(input.clean_sample_x0) == len(input.noise_epsilon) == len(input.predicted_noise)
        - 0.0 < input.alpha_bar_t < 1.0
        - 0.0 < input.beta_t < 1.0
      postconditions:
        - len(output.noisy_sample_xt) == len(input.clean_sample_x0)
        - len(output.predicted_x0) == len(input.clean_sample_x0)
        - output.noise_mse_loss >= 0.0
      certificate: "Noisy sample strictly obeys x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon"
      compatible_adapters:
        - ADAPTER-DDPM-SAMPLER
        - ADAPTER-DIFFUSION-TRAINER
      related_algos:
        - ALGO-NN-162
        - ALGO-NN-163
        - ALGO-NN-171
      references:
        - "https://arxiv.org/abs/2006.11239"
    ---
    """

    @staticmethod
    def forward(
        clean_sample_x0: List[float],
        noise_epsilon: List[float],
        predicted_noise: List[float],
        alpha_bar_t: float,
        beta_t: float,
    ) -> Dict[str, Any]:
        D = len(clean_sample_x0)
        if D == 0 or len(noise_epsilon) != D or len(predicted_noise) != D:
            raise ValueError("Precondition failed: input vector dimensions must match and be non-empty")

        if not (0.0 < alpha_bar_t < 1.0):
            raise ValueError("Precondition failed: alpha_bar_t must be in (0, 1)")
        if not (0.0 < beta_t < 1.0):
            raise ValueError("Precondition failed: beta_t must be in (0, 1)")

        sqrt_ab = math.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = math.sqrt(1.0 - alpha_bar_t)

        # 1. Forward noising: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t: List[float] = []
        for i in range(D):
            val = sqrt_ab * clean_sample_x0[i] + sqrt_one_minus_ab * noise_epsilon[i]
            x_t.append(val)

        # 2. Predicted x_0: x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        pred_x0: List[float] = []
        for i in range(D):
            val = (x_t[i] - sqrt_one_minus_ab * predicted_noise[i]) / sqrt_ab
            pred_x0.append(val)

        # 3. Training objective: MSE(epsilon, predicted_noise)
        mse_sum = 0.0
        for i in range(D):
            diff = noise_epsilon[i] - predicted_noise[i]
            mse_sum += diff * diff
        noise_mse = mse_sum / float(D)

        # 4. Reverse transition posterior mean:
        # mu_theta = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
        alpha_t = 1.0 - beta_t
        sqrt_alpha = math.sqrt(alpha_t)
        coeff_eps = beta_t / sqrt_one_minus_ab

        posterior_mean: List[float] = []
        for i in range(D):
            val = (1.0 / sqrt_alpha) * (x_t[i] - coeff_eps * predicted_noise[i])
            posterior_mean.append(val)

        return {
            "noisy_sample_xt": x_t,
            "predicted_x0": pred_x0,
            "noise_mse_loss": noise_mse,
            "posterior_mean": posterior_mean,
        }
