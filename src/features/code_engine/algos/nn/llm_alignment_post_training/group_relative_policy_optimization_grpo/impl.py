from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoGroupRelativePolicyOptimizationGrpo:
    """
    ---
    contract:
      algo_id: ALGO-NN-213
      name: NnAlgoGroupRelativePolicyOptimizationGrpo
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - grpo
        - group_relative_policy_optimization
        - critic_free_rl
        - reasoning_models
      inputs:
        type: object
        required:
          - group_rewards
          - group_policy_log_probs
          - group_old_log_probs
          - group_ref_log_probs
        properties:
          group_rewards:
            type: array
            items:
              type: number
            description: Reward scores r_1...r_G assigned to each of the G candidate generations for prompt q.
          group_policy_log_probs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Token log probabilities under current policy log pi_theta for each generation i.
          group_old_log_probs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Token log probabilities under rollout policy log pi_old for each generation i.
          group_ref_log_probs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Token log probabilities under frozen reference model log pi_ref for each generation i.
          clip_epsilon:
            type: number
            default: 0.2
            description: PPO surrogate clipping threshold epsilon in (0, 1).
          kl_coeff:
            type: number
            default: 0.04
            description: Weight coefficient beta scaling the KL divergence penalty against reference model.
      outputs:
        type: object
        required:
          - group_advantages
          - mean_group_reward
          - std_group_reward
          - grpo_loss
          - approx_kl
        properties:
          group_advantages:
            type: array
            items:
              type: number
            description: Normalized advantages A_i = (r_i - mean(r)) / std(r) for each candidate generation.
          mean_group_reward:
            type: number
            description: Empirical mean reward across the candidate group of size G.
          std_group_reward:
            type: number
            description: Empirical standard deviation of rewards across the candidate group.
          grpo_loss:
            type: number
            description: Combined GRPO objective: -mean(clipped_surrogate) + kl_coeff * approx_kl.
          approx_kl:
            type: number
            description: Mean Kullback-Leibler divergence between current policy and reference model.
      complexity:
        time: O(G * T)
        space: O(G * T)
      preconditions:
        - len(input.group_rewards) >= 2
        - len(input.group_policy_log_probs) == len(input.group_rewards)
        - len(input.group_old_log_probs) == len(input.group_rewards)
        - len(input.group_ref_log_probs) == len(input.group_rewards)
        - all(len(seq) > 0 for seq in input.group_policy_log_probs)
        - 0.0 < input.clip_epsilon < 1.0
        - input.kl_coeff >= 0.0
      postconditions:
        - len(output.group_advantages) == len(input.group_rewards)
        - output.std_group_reward >= 0.0
        - output.approx_kl >= 0.0
      certificate: "If all rewards in group are identical, group_advantages are all exactly 0.0 and clipped surrogate loss is 0.0"
      compatible_adapters:
        - ADAPTER-GRPO-TRAINER
        - ADAPTER-REASONING-ALIGNMENT-ENGINE
      related_algos:
        - ALGO-NN-200
        - ALGO-NN-210
        - ALGO-NN-214
      references:
        - "https://arxiv.org/abs/2402.03300"
    ---
    """

    @classmethod
    def forward(
        cls,
        group_rewards: List[float],
        group_policy_log_probs: List[List[float]],
        group_old_log_probs: List[List[float]],
        group_ref_log_probs: List[List[float]],
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.04,
    ) -> Dict[str, Any]:
        G = len(group_rewards)
        if G < 2:
            raise ValueError("Precondition failed: group_rewards must contain at least 2 samples (G >= 2)")
        if (
            len(group_policy_log_probs) != G
            or len(group_old_log_probs) != G
            or len(group_ref_log_probs) != G
        ):
            raise ValueError("Precondition failed: all log prob groups must match length G")
        for i in range(G):
            L = len(group_policy_log_probs[i])
            if L == 0:
                raise ValueError("Precondition failed: candidate sequence cannot be empty")
            if len(group_old_log_probs[i]) != L or len(group_ref_log_probs[i]) != L:
                raise ValueError("Precondition failed: token length mismatch in group candidate i")
        if not (0.0 < clip_epsilon < 1.0):
            raise ValueError("Precondition failed: clip_epsilon must be in (0, 1)")
        if kl_coeff < 0.0:
            raise ValueError("Precondition failed: kl_coeff must be non-negative")

        # 1. Compute group reward mean and std
        mean_r = sum(group_rewards) / float(G)
        var_r = sum((r - mean_r) ** 2 for r in group_rewards) / float(G)
        std_r = math.sqrt(var_r)

        # 2. Compute group relative advantages
        advantages: List[float] = []
        eps_denom = 1e-6
        for r in group_rewards:
            if std_r < 1e-8:
                adv = 0.0
            else:
                adv = (r - mean_r) / (std_r + eps_denom)
            advantages.append(adv)

        # 3. Compute clipped surrogate loss and KL divergence
        low = 1.0 - clip_epsilon
        high = 1.0 + clip_epsilon

        surr_group_accum = 0.0
        kl_group_accum = 0.0

        for i in range(G):
            L = len(group_policy_log_probs[i])
            adv_i = advantages[i]
            surr_seq_accum = 0.0
            kl_seq_accum = 0.0

            for t in range(L):
                lp = group_policy_log_probs[i][t]
                lp_old = group_old_log_probs[i][t]
                lp_ref = group_ref_log_probs[i][t]

                ratio = math.exp(lp - lp_old)
                clipped_ratio = max(low, min(high, ratio))

                surr = min(ratio * adv_i, clipped_ratio * adv_i)
                surr_seq_accum += surr

                # Analytic unbiased KL approximation: exp(lp_ref - lp) - (lp_ref - lp) - 1
                ratio_ref = math.exp(lp_ref - lp)
                diff_ref = lp_ref - lp
                kl_t = ratio_ref - diff_ref - 1.0
                kl_seq_accum += max(0.0, kl_t)

            surr_group_accum += (surr_seq_accum / float(L))
            kl_group_accum += (kl_seq_accum / float(L))

        mean_surr = surr_group_accum / float(G)
        mean_kl = kl_group_accum / float(G)
        grpo_loss = -mean_surr + kl_coeff * mean_kl

        return {
            "group_advantages": advantages,
            "mean_group_reward": mean_r,
            "std_group_reward": std_r,
            "grpo_loss": grpo_loss,
            "approx_kl": mean_kl,
        }
