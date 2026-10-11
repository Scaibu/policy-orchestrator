from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDirectPreferenceOptimizationDpo:
    """
    ---
    contract:
      algo_id: ALGO-NN-211
      name: NnAlgoDirectPreferenceOptimizationDpo
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - dpo
        - direct_preference_optimization
        - implicit_reward
        - bradley_terry
      inputs:
        type: object
        required:
          - policy_chosen_log_probs
          - policy_rejected_log_probs
          - ref_chosen_log_probs
          - ref_rejected_log_probs
        properties:
          policy_chosen_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of chosen responses under policy log pi_theta(y_w | x) of length B.
          policy_rejected_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of rejected responses under policy log pi_theta(y_l | x) of length B.
          ref_chosen_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of chosen responses under reference model log pi_ref(y_w | x) of length B.
          ref_rejected_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of rejected responses under reference model log pi_ref(y_l | x) of length B.
          beta:
            type: number
            default: 0.1
            description: Scaling parameter beta controlling KL constraint strength against reference model.
      outputs:
        type: object
        required:
          - dpo_loss
          - implicit_chosen_rewards
          - implicit_rejected_rewards
          - mean_implicit_reward_margin
          - preference_accuracy
        properties:
          dpo_loss:
            type: number
            description: Mean negative log-sigmoid DPO loss - (1/B) sum ln sigma(beta * (log_ratio_w - log_ratio_l)).
          implicit_chosen_rewards:
            type: array
            items:
              type: number
            description: Implicit rewards r_hat(y_w) = beta * (log pi_theta(y_w) - log pi_ref(y_w)).
          implicit_rejected_rewards:
            type: array
            items:
              type: number
            description: Implicit rewards r_hat(y_l) = beta * (log pi_theta(y_l) - log pi_ref(y_l)).
          mean_implicit_reward_margin:
            type: number
            description: Mean difference (1/B) sum (r_hat_w - r_hat_l).
          preference_accuracy:
            type: number
            description: Fraction of preference pairs where implicit reward of chosen exceeds rejected.
      complexity:
        time: O(B)
        space: O(B)
      preconditions:
        - len(input.policy_chosen_log_probs) > 0
        - len(input.policy_rejected_log_probs) == len(input.policy_chosen_log_probs)
        - len(input.ref_chosen_log_probs) == len(input.policy_chosen_log_probs)
        - len(input.ref_rejected_log_probs) == len(input.policy_chosen_log_probs)
        - input.beta > 0.0
      postconditions:
        - output.dpo_loss >= 0.0
        - len(output.implicit_chosen_rewards) == len(input.policy_chosen_log_probs)
        - len(output.implicit_rejected_rewards) == len(input.policy_chosen_log_probs)
        - 0.0 <= output.preference_accuracy <= 1.0
      certificate: "If policy matches reference model identically on all pairs, implicit rewards are 0.0 and dpo_loss is exactly ln(2)"
      compatible_adapters:
        - ADAPTER-DPO-TRAINER
        - ADAPTER-PREFERENCE-OPTIMIZER
      related_algos:
        - ALGO-NN-209
        - ALGO-NN-210
        - ALGO-NN-212
      references:
        - "https://arxiv.org/abs/2305.18290"
    ---
    """

    @classmethod
    def forward(
        cls,
        policy_chosen_log_probs: List[float],
        policy_rejected_log_probs: List[float],
        ref_chosen_log_probs: List[float],
        ref_rejected_log_probs: List[float],
        beta: float = 0.1,
    ) -> Dict[str, Any]:
        B = len(policy_chosen_log_probs)
        if B == 0:
            raise ValueError("Precondition failed: policy_chosen_log_probs must not be empty")
        if (
            len(policy_rejected_log_probs) != B
            or len(ref_chosen_log_probs) != B
            or len(ref_rejected_log_probs) != B
        ):
            raise ValueError("Precondition failed: all input log-probability arrays must have length B")
        if beta <= 0.0:
            raise ValueError("Precondition failed: beta must be positive")

        loss_accum = 0.0
        correct_count = 0
        margin_accum = 0.0

        r_chosen: List[float] = []
        r_rejected: List[float] = []

        for i in range(B):
            log_ratio_w = policy_chosen_log_probs[i] - ref_chosen_log_probs[i]
            log_ratio_l = policy_rejected_log_probs[i] - ref_rejected_log_probs[i]

            rw = beta * log_ratio_w
            rl = beta * log_ratio_l
            r_chosen.append(rw)
            r_rejected.append(rl)

            margin = rw - rl
            margin_accum += margin

            if margin > 0.0:
                correct_count += 1

            # Loss: -ln sigma(margin) = ln(1 + exp(-margin))
            if margin >= 0.0:
                loss_i = math.log1p(math.exp(-margin))
            else:
                loss_i = -margin + math.log1p(math.exp(margin))
            loss_accum += loss_i

        dpo_loss = loss_accum / B
        mean_margin = margin_accum / B
        accuracy = correct_count / float(B)

        return {
            "dpo_loss": dpo_loss,
            "implicit_chosen_rewards": r_chosen,
            "implicit_rejected_rewards": r_rejected,
            "mean_implicit_reward_margin": mean_margin,
            "preference_accuracy": accuracy,
        }
