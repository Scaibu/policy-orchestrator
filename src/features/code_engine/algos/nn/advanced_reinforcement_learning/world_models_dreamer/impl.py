from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoWorldModelsDreamer:
    """
    ---
    contract:
      algo_id: ALGO-NN-203
      name: NnAlgoWorldModelsDreamer
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - world_models
        - dreamer
        - rssm
        - latent_imagination
      inputs:
        type: object
        required:
          - observations
          - reconstructed_observations
          - rewards
          - predicted_rewards
          - posterior_means
          - posterior_logvars
          - prior_means
          - prior_logvars
          - imagined_rewards
          - imagined_values
        properties:
          observations:
            type: array
            items:
              type: number
            description: Ground truth environment observations x_t.
          reconstructed_observations:
            type: array
            items:
              type: number
            description: Decoder reconstructed observations x_hat_t.
          rewards:
            type: array
            items:
              type: number
            description: Ground truth transition scalar rewards r_t.
          predicted_rewards:
            type: array
            items:
              type: number
            description: Reward head predictions r_hat_t.
          posterior_means:
            type: array
            items:
              type: number
            description: Representation posterior means mu_post from encoder.
          posterior_logvars:
            type: array
            items:
              type: number
            description: Representation posterior log-variances log(sigma_post^2).
          prior_means:
            type: array
            items:
              type: number
            description: Dynamics transition prior means mu_prior.
          prior_logvars:
            type: array
            items:
              type: number
            description: Dynamics transition prior log-variances log(sigma_prior^2).
          imagined_rewards:
            type: array
            items:
              type: number
            description: Latent imagination horizon rollout rewards over H steps.
          imagined_values:
            type: array
            items:
              type: number
            description: Critic value predictions over imagination horizon of length H+1.
          kl_balance:
            type: number
            default: 0.8
            description: KL balancing factor between posterior and prior gradients.
          kl_scale:
            type: number
            default: 1.0
            description: Weight multiplier for latent dynamics KL divergence loss.
          gamma:
            type: number
            default: 0.99
            description: Discount factor for imagined trajectory return estimation.
          lambda_gae:
            type: number
            default: 0.95
            description: Lambda-return parameter for TD(lambda) imagination targets.
      outputs:
        type: object
        required:
          - recon_loss
          - reward_loss
          - kl_divergence
          - world_model_loss
          - lambda_returns
          - imagination_actor_loss
        properties:
          recon_loss:
            type: number
            description: Mean squared observation reconstruction error.
          reward_loss:
            type: number
            description: Mean squared reward prediction error.
          kl_divergence:
            type: number
            description: Mean analytic KL divergence between posterior and prior latents.
          world_model_loss:
            type: number
            description: Combined RSSM objective: recon_loss + reward_loss + kl_scale * kl_divergence.
          lambda_returns:
            type: array
            items:
              type: number
            description: Generalized TD(lambda) return targets along imagination trajectory.
          imagination_actor_loss:
            type: number
            description: Policy loss maximizing expected imagined lambda returns.
      complexity:
        time: O(T + H)
        space: O(T + H)
      preconditions:
        - len(input.observations) > 0
        - len(input.reconstructed_observations) == len(input.observations)
        - len(input.rewards) == len(input.observations)
        - len(input.predicted_rewards) == len(input.observations)
        - len(input.posterior_means) == len(input.observations)
        - len(input.posterior_logvars) == len(input.observations)
        - len(input.prior_means) == len(input.observations)
        - len(input.prior_logvars) == len(input.observations)
        - len(input.imagined_values) == len(input.imagined_rewards) + 1
        - 0.0 <= input.kl_balance <= 1.0
        - 0.0 < input.gamma <= 1.0
        - 0.0 <= input.lambda_gae <= 1.0
      postconditions:
        - len(output.lambda_returns) == len(input.imagined_rewards)
        - output.recon_loss >= 0.0
        - output.reward_loss >= 0.0
        - output.kl_divergence >= 0.0
      certificate: "If posterior equals prior exactly and predictions match targets, world_model_loss is exactly 0.0"
      compatible_adapters:
        - ADAPTER-MODEL-BASED-RL
        - ADAPTER-RSSM-WORLD-TRAINER
      related_algos:
        - ALGO-NN-201
        - ALGO-NN-202
      references:
        - "https://arxiv.org/abs/1912.01603"
    ---
    """

    @classmethod
    def forward(
        cls,
        observations: List[float],
        reconstructed_observations: List[float],
        rewards: List[float],
        predicted_rewards: List[float],
        posterior_means: List[float],
        posterior_logvars: List[float],
        prior_means: List[float],
        prior_logvars: List[float],
        imagined_rewards: List[float],
        imagined_values: List[float],
        kl_balance: float = 0.8,
        kl_scale: float = 1.0,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
    ) -> Dict[str, Any]:
        T = len(observations)
        if T == 0:
            raise ValueError("Precondition failed: observations must not be empty")
        if (
            len(reconstructed_observations) != T
            or len(rewards) != T
            or len(predicted_rewards) != T
            or len(posterior_means) != T
            or len(posterior_logvars) != T
            or len(prior_means) != T
            or len(prior_logvars) != T
        ):
            raise ValueError("Precondition failed: all sequence inputs must have length T")
        H = len(imagined_rewards)
        if len(imagined_values) != H + 1:
            raise ValueError("Precondition failed: imagined_values length must be len(imagined_rewards) + 1")
        if not (0.0 <= kl_balance <= 1.0):
            raise ValueError("Precondition failed: kl_balance must be in [0, 1]")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("Precondition failed: gamma must be in (0, 1]")
        if not (0.0 <= lambda_gae <= 1.0):
            raise ValueError("Precondition failed: lambda_gae must be in [0, 1]")

        recon_err = 0.0
        reward_err = 0.0
        kl_accum = 0.0

        for t in range(T):
            d_x = observations[t] - reconstructed_observations[t]
            recon_err += d_x * d_x

            d_r = rewards[t] - predicted_rewards[t]
            reward_err += d_r * d_r

            mu_q, lv_q = posterior_means[t], posterior_logvars[t]
            mu_p, lv_p = prior_means[t], prior_logvars[t]
            var_q = math.exp(lv_q)
            var_p = math.exp(lv_p)

            # KL(q || p) for 1D Gaussians: 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p)^2) / var_p - 1.0)
            kl_t = 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
            kl_accum += max(0.0, kl_t)

        recon_loss = recon_err / T
        reward_loss = reward_err / T
        mean_kl = kl_accum / T
        world_model_loss = recon_loss + reward_loss + kl_scale * mean_kl

        # Compute TD(lambda) returns along imagination trajectory backwards
        # R_t^lambda = r_t + gamma * ((1 - lambda) * V(s_{t+1}) + lambda * R_{t+1}^lambda)
        lambda_returns = [0.0] * H
        next_ret = imagined_values[H]
        for t in reversed(range(H)):
            ret = imagined_rewards[t] + gamma * (
                (1.0 - lambda_gae) * imagined_values[t + 1] + lambda_gae * next_ret
            )
            lambda_returns[t] = ret
            next_ret = ret

        # Actor loss in imagination: maximize expected lambda returns
        imagination_actor_loss = - (sum(lambda_returns) / H) if H > 0 else 0.0

        return {
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "kl_divergence": mean_kl,
            "world_model_loss": world_model_loss,
            "lambda_returns": lambda_returns,
            "imagination_actor_loss": imagination_actor_loss,
        }
