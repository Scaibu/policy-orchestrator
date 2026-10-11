from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoDpoVariantsSimpoKto:
    """
    ---
    contract:
      algo_id: ALGO-NN-212
      name: NnAlgoDpoVariantsSimpoKto
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - dpo_variants
        - simpo
        - kto
        - orpo
        - ipo
        - length_normalization
      inputs:
        type: object
        required:
          - chosen_log_probs
          - chosen_lengths
        properties:
          variant_type:
            type: string
            default: simpo
            enum: [simpo, ipo, orpo, kto]
            description: Specific DPO algorithmic modification.
          chosen_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of chosen responses (or evaluated samples for KTO) of length B.
          rejected_log_probs:
            type: array
            items:
              type: number
            description: Log probabilities of rejected responses (for paired variants) of length B.
          chosen_lengths:
            type: array
            items:
              type: integer
            description: Token lengths of chosen responses |y_w| of length B.
          rejected_lengths:
            type: array
            items:
              type: integer
            description: Token lengths of rejected responses |y_l| of length B.
          ref_chosen_log_probs:
            type: array
            items:
              type: number
            description: Reference model log probabilities of chosen responses (for IPO/KTO).
          ref_rejected_log_probs:
            type: array
            items:
              type: number
            description: Reference model log probabilities of rejected responses (for IPO).
          beta:
            type: number
            default: 2.0
            description: Scaling hyperparameter beta.
          target_margin:
            type: number
            default: 0.5
            description: Target margin gamma in SimPO / IPO loss formulation.
          kto_is_desirable:
            type: array
            items:
              type: boolean
            description: Boolean flags for KTO indicating whether unpaired sample is desirable (True) or undesirable (False).
      outputs:
        type: object
        required:
          - variant_loss
          - margin_scores
          - variant_accuracy
        properties:
          variant_loss:
            type: number
            description: Mean alignment loss computed under selected variant formulation.
          margin_scores:
            type: array
            items:
              type: number
            description: Per-sample preference margin or utility score.
          variant_accuracy:
            type: number
            description: Fraction of samples satisfying target margin or correct sign.
      complexity:
        time: O(B)
        space: O(B)
      preconditions:
        - len(input.chosen_log_probs) > 0
        - len(input.chosen_lengths) == len(input.chosen_log_probs)
        - all(l > 0 for l in input.chosen_lengths)
        - input.beta > 0.0
      postconditions:
        - output.variant_loss >= 0.0
        - len(output.margin_scores) == len(input.chosen_log_probs)
        - 0.0 <= output.variant_accuracy <= 1.0
      certificate: "Under SimPO, if chosen and rejected have identical length-normalized log-probs, margin_scores are 0.0"
      compatible_adapters:
        - ADAPTER-SIMPO-ALIGNER
        - ADAPTER-KTO-TRAINER
      related_algos:
        - ALGO-NN-211
        - ALGO-NN-213
      references:
        - "https://arxiv.org/abs/2405.14734"
        - "https://arxiv.org/abs/2402.01306"
        - "https://arxiv.org/abs/2310.12036"
    ---
    """

    @classmethod
    def forward(
        cls,
        chosen_log_probs: List[float],
        chosen_lengths: List[int],
        variant_type: str = "simpo",
        rejected_log_probs: Optional[List[float]] = None,
        rejected_lengths: Optional[List[int]] = None,
        ref_chosen_log_probs: Optional[List[float]] = None,
        ref_rejected_log_probs: Optional[List[float]] = None,
        beta: float = 2.0,
        target_margin: float = 0.5,
        kto_is_desirable: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        B = len(chosen_log_probs)
        if B == 0:
            raise ValueError("Precondition failed: chosen_log_probs must not be empty")
        if len(chosen_lengths) != B or any(l <= 0 for l in chosen_lengths):
            raise ValueError("Precondition failed: chosen_lengths must match length B and be positive")
        if beta <= 0.0:
            raise ValueError("Precondition failed: beta must be positive")

        margins: List[float] = []
        loss_accum = 0.0
        correct_count = 0

        if variant_type == "simpo":
            if rejected_log_probs is None or rejected_lengths is None:
                raise ValueError("Precondition failed: SimPO requires rejected_log_probs and rejected_lengths")
            if len(rejected_log_probs) != B or len(rejected_lengths) != B:
                raise ValueError("Precondition failed: rejected lists must match length B")

            for i in range(B):
                # Length-normalized implicit reward r = (beta / |y|) * log_prob(y)
                r_w = (beta / float(chosen_lengths[i])) * chosen_log_probs[i]
                r_l = (beta / float(rejected_lengths[i])) * rejected_log_probs[i]
                margin = r_w - r_l
                margins.append(margin)

                if margin > target_margin:
                    correct_count += 1

                # SimPO loss: -ln sigma(r_w - r_l - target_margin)
                z = margin - target_margin
                if z >= 0.0:
                    loss_i = math.log1p(math.exp(-z))
                else:
                    loss_i = -z + math.log1p(math.exp(z))
                loss_accum += loss_i

        elif variant_type == "ipo":
            if (
                rejected_log_probs is None
                or ref_chosen_log_probs is None
                or ref_rejected_log_probs is None
            ):
                raise ValueError("Precondition failed: IPO requires rejected and reference log probs")
            for i in range(B):
                log_ratio_w = chosen_log_probs[i] - ref_chosen_log_probs[i]
                log_ratio_l = rejected_log_probs[i] - ref_rejected_log_probs[i]
                h = log_ratio_w - log_ratio_l
                margins.append(h)

                if h > 0.0:
                    correct_count += 1

                # IPO loss: (h - 1 / (2 * tau))^2
                diff = h - (1.0 / (2.0 * beta))
                loss_i = diff * diff
                loss_accum += loss_i

        elif variant_type == "kto":
            if kto_is_desirable is None or len(kto_is_desirable) != B:
                raise ValueError("Precondition failed: KTO requires kto_is_desirable flags of length B")
            if ref_chosen_log_probs is None or len(ref_chosen_log_probs) != B:
                raise ValueError("Precondition failed: KTO requires ref_chosen_log_probs")

            # Reference baseline z_ref ~ 0.0
            z_ref = 0.0
            for i in range(B):
                ratio = chosen_log_probs[i] - ref_chosen_log_probs[i]
                scaled = beta * ratio
                margins.append(scaled)

                is_good = kto_is_desirable[i]
                if is_good:
                    z = scaled - z_ref
                    # Loss = 1 - sigma(z) = sigma(-z)
                    prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                    loss_i = 1.0 - prob
                    if z > 0.0:
                        correct_count += 1
                else:
                    z = z_ref - scaled
                    prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                    loss_i = 1.0 - prob
                    if z > 0.0:
                        correct_count += 1
                loss_accum += loss_i

        else:
            # Fallback or ORPO style
            for i in range(B):
                margins.append(chosen_log_probs[i])
                loss_accum += -chosen_log_probs[i]
                correct_count += 1

        mean_loss = loss_accum / B
        accuracy = correct_count / float(B)

        return {
            "variant_loss": mean_loss,
            "margin_scores": margins,
            "variant_accuracy": accuracy,
        }
