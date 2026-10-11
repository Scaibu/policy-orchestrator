from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoWassersteinGanGp:
    """
    ---
    contract:
      algo_id: ALGO-NN-155
      name: NnAlgoWassersteinGanGp
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - wgan_gp
        - earth_movers_distance
        - gradient_penalty
      inputs:
        type: object
        required:
          - critic_real_scores
          - critic_fake_scores
          - interpolated_gradients
        properties:
          critic_real_scores:
            type: array
            items:
              type: number
            description: Unbounded critic scalar outputs D(x) on real batch.
          critic_fake_scores:
            type: array
            items:
              type: number
            description: Unbounded critic scalar outputs D(G(z)) on generated batch.
          interpolated_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Gradients of critic with respect to interpolated points grad_x D(x_hat).
          lambda_gp:
            type: number
            default: 10.0
            description: Gradient penalty regularization weight.
      outputs:
        type: object
        required:
          - critic_loss
          - generator_loss
          - wasserstein_distance
          - gradient_penalty
        properties:
          critic_loss:
            type: number
            description: Total penalized critic objective to minimize.
          generator_loss:
            type: number
            description: Generator objective -E[D(G(z))].
          wasserstein_distance:
            type: number
            description: Estimated Earth Mover's Distance E[D(x)] - E[D(G(z))].
          gradient_penalty:
            type: number
            description: 1-Lipschitz gradient norm penalty E[(||grad||_2 - 1)^2].
      parameters: {}
      input_assumptions:
        - critic_real_scores and critic_fake_scores are non-empty
        - interpolated_gradients has non-empty vectors with uniform dimension
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact floating-point computation"
      uses_model: false
      complexity:
        variables:
          N_interp: number of interpolated points
          d: feature dimension
        time_worst: O(N_interp * d)
        time_typical: O(N_interp * d)
        space: O(1)
      preconditions:
        - len(input.critic_real_scores) > 0 and len(input.critic_fake_scores) > 0
        - len(input.interpolated_gradients) > 0 and len(input.interpolated_gradients[0]) > 0
        - input.lambda_gp >= 0.0
      postconditions:
        - output.gradient_penalty >= 0.0
      certificate: "Critic loss equals -wasserstein_distance + lambda_gp * gradient_penalty"
      compatible_adapters:
        - ADAPTER-STABLE-GAN-TRAINER
      related_algos:
        - ALGO-NN-154
        - ALGO-NN-156
      references:
        - "https://arxiv.org/abs/1704.00028"
    ---
    """

    @staticmethod
    def forward(
        critic_real_scores: List[float],
        critic_fake_scores: List[float],
        interpolated_gradients: List[List[float]],
        lambda_gp: float = 10.0,
    ) -> Dict[str, Any]:
        if len(critic_real_scores) == 0 or len(critic_fake_scores) == 0:
            raise ValueError("Precondition failed: critic scores cannot be empty")
        if len(interpolated_gradients) == 0 or len(interpolated_gradients[0]) == 0:
            raise ValueError("Precondition failed: interpolated_gradients cannot be empty")
        if lambda_gp < 0.0:
            raise ValueError("Precondition failed: lambda_gp must be non-negative")

        mean_real = sum(critic_real_scores) / float(len(critic_real_scores))
        mean_fake = sum(critic_fake_scores) / float(len(critic_fake_scores))

        wasserstein_dist = mean_real - mean_fake
        gen_loss = -mean_fake

        # Gradient penalty: E[(||grad||_2 - 1)^2]
        gp_sum = 0.0
        for grad in interpolated_gradients:
            norm_sq = sum(g * g for g in grad)
            norm = math.sqrt(norm_sq)
            diff = norm - 1.0
            gp_sum += diff * diff

        gradient_penalty = gp_sum / float(len(interpolated_gradients))
        critic_loss = -wasserstein_dist + lambda_gp * gradient_penalty

        return {
            "critic_loss": critic_loss,
            "generator_loss": gen_loss,
            "wasserstein_distance": wasserstein_dist,
            "gradient_penalty": gradient_penalty,
        }
