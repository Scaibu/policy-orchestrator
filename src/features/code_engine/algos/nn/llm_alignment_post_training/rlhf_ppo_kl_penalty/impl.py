from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRlhfPpoKlPenalty:
    """
    ---
    contract:
      algo_id: ALGO-NN-210
      name: NnAlgoRlhfPpoKlPenalty
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - rlhf
        - ppo
        - kl_penalty
        - gae
      inputs:
        type: object
        required:
          - policy_log_probs
          - old_log_probs
          - ref_log_probs
          - value_estimates
          - sequence_reward
        properties:
          policy_log_probs:
            type: array
            items:
              type: number
            description: Current policy token log probabilities log pi_theta(y_t | x, y_<t) of length T.
          old_log_probs:
            type: array
            items:
              type: number
            description: Rollout policy token log probabilities log pi_old(y_t | x, y_<t) of length T.
          ref_log_probs:
            type: array
            items:
              type: number
            description: Frozen base reference model token log probabilities log pi_ref of length T.
          value_estimates:
            type: array
            items:
              type: number
            description: Critic token value predictions V(s_t) of length T.
          sequence_reward:
            type: number
            description: Scalar score R(x, y) from reward model assigned at final token.
          kl_penalty_coef:
            type: number
            default: 0.05
            description: Weight beta_KL scaling token-level divergence penalty.
          gamma:
            type: number
            default: 1.0
            description: Return discount factor gamma (typically 1.0 for episodic text generation).
          lambda_gae:
            type: number
            default: 0.95
            description: Generalized Advantage Estimation decay factor lambda.
          clip_epsilon:
            type: number
            default: 0.2
            description: PPO surrogate clipping threshold epsilon.
      outputs:
        type: object
        required:
          - penalized_rewards
          - total_kl_divergence
          - ppo_policy_loss
          - value_loss
          - mean_advantage
        properties:
          penalized_rewards:
            type: array
            items:
              type: number
            description: Per-token reward trajectory r_t = -beta * KL_t, with sequence_reward added at step T.
          total_kl_divergence:
            type: number
            description: Sum of per-token KL divergences sum (log pi_theta - log pi_ref) across generation.
          ppo_policy_loss:
            type: number
            description: PPO clipped surrogate objective loss over token trajectory.
          value_loss:
            type: number
            description: Mean squared error between value estimates and GAE return targets.
          mean_advantage:
            type: number
            description: Mean estimated token advantage across sequence.
      complexity:
        time: O(T)
        space: O(T)
      preconditions:
        - len(input.policy_log_probs) > 0
        - len(input.old_log_probs) == len(input.policy_log_probs)
        - len(input.ref_log_probs) == len(input.policy_log_probs)
        - len(input.value_estimates) == len(input.policy_log_probs)
        - input.kl_penalty_coef >= 0.0
        - 0.0 < input.clip_epsilon < 1.0
      postconditions:
        - len(output.penalized_rewards) == len(input.policy_log_probs)
      certificate: "If policy matches reference model exactly, total_kl_divergence is 0.0 and sum(penalized_rewards) equals sequence_reward"
      compatible_adapters:
        - ADAPTER-RLHF-ALIGNMENT-TRAINER
        - ADAPTER-PPO-KL-REGULATOR
      related_algos:
        - ALGO-NN-200
        - ALGO-NN-209
        - ALGO-NN-211
      references:
        - "https://arxiv.org/abs/2009.01325"
        - "https://arxiv.org/abs/2203.02155"
    ---
    """

    @classmethod
    def forward(
        cls,
        policy_log_probs: List[float],
        old_log_probs: List[float],
        ref_log_probs: List[float],
        value_estimates: List[float],
        sequence_reward: float,
        kl_penalty_coef: float = 0.05,
        gamma: float = 1.0,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
    ) -> Dict[str, Any]:
        T = len(policy_log_probs)
        if T == 0:
            raise ValueError("Precondition failed: policy_log_probs must not be empty")
        if (
            len(old_log_probs) != T
            or len(ref_log_probs) != T
            or len(value_estimates) != T
        ):
            raise ValueError("Precondition failed: all input sequence lists must match length T")
        if kl_penalty_coef < 0.0:
            raise ValueError("Precondition failed: kl_penalty_coef must be non-negative")
        if not (0.0 < clip_epsilon < 1.0):
            raise ValueError("Precondition failed: clip_epsilon must be in (0, 1)")

        # 1. Compute per-token KL and penalized rewards
        penalized_rewards: List[float] = []
        total_kl = 0.0
        for t in range(T):
            kl_t = policy_log_probs[t] - ref_log_probs[t]
            total_kl += kl_t
            r_t = -kl_penalty_coef * kl_t
            if t == T - 1:
                r_t += sequence_reward
            penalized_rewards.append(r_t)

        # 2. GAE computation: delta_t = r_t + gamma * V_{t+1} - V_t
        advantages = [0.0] * T
        returns = [0.0] * T
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = value_estimates[t + 1] if t + 1 < T else 0.0
            delta_t = penalized_rewards[t] + gamma * next_val - value_estimates[t]
            last_gae = delta_t + gamma * lambda_gae * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + value_estimates[t]

        # 3. PPO clipped loss
        surr_accum = 0.0
        val_err_accum = 0.0
        low = 1.0 - clip_epsilon
        high = 1.0 + clip_epsilon

        for t in range(T):
            ratio = math.exp(policy_log_probs[t] - old_log_probs[t])
            clipped_ratio = max(low, min(high, ratio))
            adv = advantages[t]
            surr = min(ratio * adv, clipped_ratio * adv)
            surr_accum += surr

            diff_v = value_estimates[t] - returns[t]
            val_err_accum += diff_v * diff_v

        ppo_loss = -surr_accum / T
        value_loss = val_err_accum / T
        mean_adv = sum(advantages) / T

        return {
            "penalized_rewards": penalized_rewards,
            "total_kl_divergence": total_kl,
            "ppo_policy_loss": ppo_loss,
            "value_loss": value_loss,
            "mean_advantage": mean_adv,
        }
