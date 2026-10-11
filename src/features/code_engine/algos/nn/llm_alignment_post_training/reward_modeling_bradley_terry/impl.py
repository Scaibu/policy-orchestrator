from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoRewardModelingBradleyTerry:
    """
    ---
    contract:
      algo_id: ALGO-NN-209
      name: NnAlgoRewardModelingBradleyTerry
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - reward_modeling
        - bradley_terry
        - preference_learning
        - length_bias
      inputs:
        type: object
        required:
          - chosen_rewards
          - rejected_rewards
        properties:
          chosen_rewards:
            type: array
            items:
              type: number
            description: Scalar rewards r(x, y_w) assigned to preferred responses of length B.
          rejected_rewards:
            type: array
            items:
              type: number
            description: Scalar rewards r(x, y_l) assigned to rejected responses of length B.
          chosen_lengths:
            type: array
            items:
              type: integer
            description: Token lengths of chosen responses for length bias diagnosis.
          rejected_lengths:
            type: array
            items:
              type: integer
            description: Token lengths of rejected responses for length bias diagnosis.
          target_margin:
            type: number
            default: 0.0
            description: Minimum target reward margin m in -ln sigma((r_w - r_l) - m).
      outputs:
        type: object
        required:
          - bradley_terry_loss
          - pairwise_accuracy
          - mean_reward_margin
          - length_bias_correlation
        properties:
          bradley_terry_loss:
            type: number
            description: Mean negative log-sigmoid loss - (1/B) sum ln sigma(r_w - r_l - m).
          pairwise_accuracy:
            type: number
            description: Fraction of pairs where chosen response scored higher than rejected response.
          mean_reward_margin:
            type: number
            description: Empirical mean difference (1/B) sum (r_w - r_l).
          length_bias_correlation:
            type: number
            description: Pearson correlation between reward difference (r_w - r_l) and length difference (len_w - len_l).
      complexity:
        time: O(B)
        space: O(B)
      preconditions:
        - len(input.chosen_rewards) > 0
        - len(input.rejected_rewards) == len(input.chosen_rewards)
        - input.target_margin >= 0.0
      postconditions:
        - output.bradley_terry_loss >= 0.0
        - 0.0 <= output.pairwise_accuracy <= 1.0
        - -1.0 <= output.length_bias_correlation <= 1.0
      certificate: "If all chosen rewards exceed rejected rewards by at least 10.0, pairwise_accuracy is 1.0 and loss is near 0.0"
      compatible_adapters:
        - ADAPTER-REWARD-MODEL-TRAINER
        - ADAPTER-RLHF-ALIGNMENT-PIPELINE
      related_algos:
        - ALGO-NN-015
        - ALGO-NN-210
        - ALGO-NN-211
      references:
        - "https://arxiv.org/abs/2203.02155"
        - "https://doi.org/10.2307/2334029"
    ---
    """

    @classmethod
    def forward(
        cls,
        chosen_rewards: List[float],
        rejected_rewards: List[float],
        chosen_lengths: Optional[List[int]] = None,
        rejected_lengths: Optional[List[int]] = None,
        target_margin: float = 0.0,
    ) -> Dict[str, Any]:
        B = len(chosen_rewards)
        if B == 0:
            raise ValueError("Precondition failed: chosen_rewards must not be empty")
        if len(rejected_rewards) != B:
            raise ValueError("Precondition failed: rejected_rewards length must match chosen_rewards")
        if target_margin < 0.0:
            raise ValueError("Precondition failed: target_margin must be non-negative")

        loss_accum = 0.0
        correct_count = 0
        margin_accum = 0.0

        diff_rewards: List[float] = []
        diff_lengths: List[float] = []

        track_lengths = (
            chosen_lengths is not None
            and rejected_lengths is not None
            and len(chosen_lengths) == B
            and len(rejected_lengths) == B
        )

        for i in range(B):
            r_w = chosen_rewards[i]
            r_l = rejected_rewards[i]
            diff_r = r_w - r_l
            margin_accum += diff_r
            diff_rewards.append(diff_r)

            if diff_r > 0.0:
                correct_count += 1

            # Softplus loss: -ln sigma(z) = ln(1 + exp(-z))
            z = diff_r - target_margin
            if z >= 0.0:
                loss_i = math.log1p(math.exp(-z))
            else:
                loss_i = -z + math.log1p(math.exp(z))
            loss_accum += loss_i

            if track_lengths:
                diff_len = float(chosen_lengths[i] - rejected_lengths[i])
                diff_lengths.append(diff_len)

        mean_loss = loss_accum / B
        accuracy = correct_count / float(B)
        mean_margin = margin_accum / B

        # Compute Pearson correlation between reward diff and length diff
        if track_lengths and len(diff_lengths) == B:
            mean_dl = sum(diff_lengths) / B
            cov = sum((diff_rewards[i] - mean_margin) * (diff_lengths[i] - mean_dl) for i in range(B)) / B
            var_r = sum((diff_rewards[i] - mean_margin) ** 2 for i in range(B)) / B
            var_l = sum((diff_lengths[i] - mean_dl) ** 2 for i in range(B)) / B
            denom = math.sqrt(var_r * var_l)
            if denom > 1e-9:
                length_corr = max(-1.0, min(1.0, cov / denom))
            else:
                length_corr = 0.0
        else:
            length_corr = 0.0

        return {
            "bradley_terry_loss": mean_loss,
            "pairwise_accuracy": accuracy,
            "mean_reward_margin": mean_margin,
            "length_bias_correlation": length_corr,
        }
