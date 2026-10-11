from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoProximalPolicyOptimizationPpo:
    """
    ---
    contract:
      algo_id: ALGO-NN-200
      name: NnAlgoProximalPolicyOptimizationPpo
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - ppo
        - proximal_policy_optimization
        - clipped_surrogate
        - trust_region
      inputs:
        type: object
        required:
          - current_log_probs
          - old_log_probs
          - advantages
        properties:
          current_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities under current policy log pi_theta(a_t | s_t) of length T.
          old_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities under rollout policy log pi_old(a_t | s_t) of length T.
          advantages:
            type: array
            items:
              type: number
            description: Estimated advantages A_t of length T.
          clip_epsilon:
            type: number
            default: 0.2
            minimum: 0.001
            maximum: 0.5
            description: PPO clipping threshold parameter epsilon.
      outputs:
        type: object
        required:
          - ppo_policy_loss
          - probability_ratios
          - clip_fraction
          - approximate_kl
        properties:
          ppo_policy_loss:
            type: number
            description: Clipped surrogate objective loss - (1/T) sum min(r * A, clip(r) * A).
          probability_ratios:
            type: array
            items:
              type: number
            description: Importance sampling ratios r_t = pi_theta / pi_old of length T.
          clip_fraction:
            type: number
            minimum: 0.0
            maximum: 1.0
            description: Proportion of transitions where ratio r_t was clipped outside [1 - eps, 1 + eps].
          approximate_kl:
            type: number
            description: Empirical mean approximate KL divergence (1/T) sum (log_old - log_current).
      complexity:
        time: O(T)
        space: O(T)
      preconditions:
        - len(input.current_log_probs) > 0
        - len(input.old_log_probs) == len(input.current_log_probs)
        - len(input.advantages) == len(input.current_log_probs)
        - 0.0 < input.clip_epsilon < 1.0
      postconditions:
        - len(output.probability_ratios) == len(input.current_log_probs)
        - 0.0 <= output.clip_fraction <= 1.0
      certificate: "If current_log_probs equals old_log_probs, probability_ratios are all exactly 1.0 and clip_fraction is 0.0"
      compatible_adapters:
        - ADAPTER-PPO-EPOCH-TRAINER
        - ADAPTER-RLHF-POLICY-ALIGNER
      related_algos:
        - ALGO-NN-198
        - ALGO-NN-199
      references:
        - "https://arxiv.org/abs/1707.06347"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_log_probs: List[float],
        old_log_probs: List[float],
        advantages: List[float],
        clip_epsilon: float = 0.2,
    ) -> Dict[str, Any]:
        T = len(current_log_probs)
        if T == 0:
            raise ValueError("Precondition failed: current_log_probs cannot be empty")
        if len(old_log_probs) != T or len(advantages) != T:
            raise ValueError("Precondition failed: input arrays must have identical length T")
        if not (0.0 < clip_epsilon < 1.0):
            raise ValueError("Precondition failed: clip_epsilon must be in (0, 1)")

        ratios: List[float] = []
        clipped_count = 0
        surrogate_accum = 0.0
        kl_accum = 0.0

        lower_bound = 1.0 - clip_epsilon
        upper_bound = 1.0 + clip_epsilon

        for t in range(T):
            cur_lp = current_log_probs[t]
            old_lp = old_log_probs[t]
            adv = advantages[t]

            # 1. Ratio: r_t = exp(cur_lp - old_lp)
            log_ratio = cur_lp - old_lp
            ratio = math.exp(log_ratio)
            ratios.append(ratio)

            # 2. Approximate KL: (old_lp - cur_lp)
            kl_accum += (old_lp - cur_lp)

            # 3. Clipped ratio
            clipped_ratio = max(lower_bound, min(upper_bound, ratio))
            if ratio < lower_bound or ratio > upper_bound:
                clipped_count += 1

            # 4. PPO clipped surrogate: min(r * A, clip(r) * A)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            surrogate_accum += min(surr1, surr2)

        # Policy loss to minimize is negative of surrogate objective
        ppo_loss = - (surrogate_accum / float(T))
        clip_frac = float(clipped_count) / float(T)
        approx_kl = max(0.0, kl_accum / float(T))

        return {
            "ppo_policy_loss": ppo_loss,
            "probability_ratios": ratios,
            "clip_fraction": clip_frac,
            "approximate_kl": approx_kl,
        }
