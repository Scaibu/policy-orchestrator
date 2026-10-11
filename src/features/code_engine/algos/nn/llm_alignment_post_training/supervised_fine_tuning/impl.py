from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSupervisedFineTuning:
    """
    ---
    contract:
      algo_id: ALGO-NN-208
      name: NnAlgoSupervisedFineTuning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - supervised_fine_tuning
        - sft
        - instruction_tuning
        - prompt_masking
      inputs:
        type: object
        required:
          - target_tokens
          - predicted_logits
          - loss_mask
        properties:
          target_tokens:
            type: array
            items:
              type: integer
            description: Ground truth target token vocabulary IDs of length T.
          predicted_logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Model unnormalized output logits across vocabulary of shape (T, V).
          loss_mask:
            type: array
            items:
              type: integer
            description: Binary mask where 0 indicates prompt token (ignored) and 1 indicates assistant response token.
          label_smoothing:
            type: number
            default: 0.0
            description: Label smoothing factor epsilon in [0, 1).
      outputs:
        type: object
        required:
          - masked_sft_loss
          - completion_perplexity
          - active_token_count
          - token_accuracy
        properties:
          masked_sft_loss:
            type: number
            description: Mean next-token cross-entropy loss computed strictly over assistant tokens.
          completion_perplexity:
            type: number
            description: Exponential of masked loss exp(loss) measuring completion perplexity.
          active_token_count:
            type: integer
            description: Total number of supervised assistant tokens with mask 1.
          token_accuracy:
            type: number
            description: Fraction of completion tokens where argmax logit matched target token ID.
      complexity:
        time: O(T * V)
        space: O(T * V)
      preconditions:
        - len(input.target_tokens) > 0
        - len(input.predicted_logits) == len(input.target_tokens)
        - len(input.loss_mask) == len(input.target_tokens)
        - any(m == 1 for m in input.loss_mask)
        - 0.0 <= input.label_smoothing < 1.0
      postconditions:
        - output.masked_sft_loss >= 0.0
        - output.completion_perplexity >= 1.0
        - output.active_token_count > 0
        - 0.0 <= output.token_accuracy <= 1.0
      certificate: "If predicted logits assign probability 1.0 to all target tokens, masked_sft_loss is 0.0 and token_accuracy is 1.0"
      compatible_adapters:
        - ADAPTER-SFT-TRAINER
        - ADAPTER-INSTRUCTION-FINE-TUNER
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-207
      references:
        - "https://arxiv.org/abs/2109.01652"
        - "https://arxiv.org/abs/2203.02155"
    ---
    """

    @classmethod
    def forward(
        cls,
        target_tokens: List[int],
        predicted_logits: List[List[float]],
        loss_mask: List[int],
        label_smoothing: float = 0.0,
    ) -> Dict[str, Any]:
        T = len(target_tokens)
        if T == 0:
            raise ValueError("Precondition failed: target_tokens must not be empty")
        if len(predicted_logits) != T or len(loss_mask) != T:
            raise ValueError("Precondition failed: predicted_logits and loss_mask must match target_tokens length T")
        if not any(m == 1 for m in loss_mask):
            raise ValueError("Precondition failed: loss_mask must contain at least one supervised completion token (1)")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("Precondition failed: label_smoothing must be in [0, 1)")

        active_count = 0
        correct_count = 0
        total_loss_accum = 0.0

        for t in range(T):
            if loss_mask[t] != 1:
                continue

            active_count += 1
            logits = predicted_logits[t]
            V = len(logits)
            if V == 0:
                raise ValueError("Precondition failed: vocabulary dimension V must be positive")

            target_id = target_tokens[t]
            if not (0 <= target_id < V):
                raise ValueError("Precondition failed: target token ID must be in valid vocabulary range [0, V-1]")

            # Numerically stable Log-Softmax
            max_l = max(logits)
            sum_exp = sum(math.exp(l - max_l) for l in logits)
            log_z = max_l + math.log(sum_exp)
            log_probs = [l - log_z for l in logits]

            # Accuracy check
            argmax_id = max(range(V), key=lambda i: logits[i])
            if argmax_id == target_id:
                correct_count += 1

            # Cross entropy with optional label smoothing
            if label_smoothing == 0.0:
                loss_t = -log_probs[target_id]
            else:
                smooth_val = label_smoothing / V
                loss_t = - ((1.0 - label_smoothing) * log_probs[target_id] + smooth_val * sum(log_probs))

            total_loss_accum += loss_t

        mean_loss = total_loss_accum / active_count
        ppl = math.exp(min(100.0, mean_loss))
        accuracy = correct_count / float(active_count)

        return {
            "masked_sft_loss": mean_loss,
            "completion_perplexity": ppl,
            "active_token_count": active_count,
            "token_accuracy": accuracy,
        }
