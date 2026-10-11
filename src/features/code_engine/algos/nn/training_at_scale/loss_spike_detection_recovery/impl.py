from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLossSpikeDetectionRecovery:
    """
    ---
    contract:
      algo_id: ALGO-NN-242
      name: NnAlgoLossSpikeDetectionRecovery
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - loss_spike_detection
        - fault_recovery
        - rolling_zscore
        - adaptive_learning_rate
      inputs:
        type: object
        required:
          - current_loss
          - current_grad_norm
          - loss_history
        properties:
          current_loss:
            type: number
            description: Training loss evaluated at the current optimization step.
          current_grad_norm:
            type: number
            description: L2 norm of the unclipped gradient vector at the current step.
          loss_history:
            type: array
            items:
              type: number
            description: Rolling window of preceding training loss values.
          threshold_std_devs:
            type: number
            default: 3.0
            description: Number of rolling standard deviations above mean qualifying as an anomaly.
          learning_rate:
            type: number
            default: 0.0001
            description: Base optimizer learning rate.
          grad_norm_cap:
            type: number
            default: 10.0
            description: Threshold on gradient norm indicating explosive gradient divergence.
      outputs:
        type: object
        required:
          - is_spike_detected
          - recommended_action
          - recommended_learning_rate
          - z_score
          - rolling_mean
          - rolling_std
        properties:
          is_spike_detected:
            type: boolean
            description: True if current loss or gradient norm exceeds statistical bounds.
          recommended_action:
            type: string
            enum: [proceed, rollback_and_skip, throttle_lr]
            description: Automated recovery action to preserve training stability.
          recommended_learning_rate:
            type: number
            description: Adjusted learning rate recommended for subsequent optimization step.
          z_score:
            type: number
            description: Statistical deviation (current_loss - mean) / std.
          rolling_mean:
            type: number
            description: Mean loss over historical rolling window.
          rolling_std:
            type: number
            description: Standard deviation of loss over historical window.
      complexity:
        time: O(len(loss_history))
        space: O(1)
      preconditions:
        - input.threshold_std_devs > 0.0
        - input.learning_rate > 0.0
        - input.grad_norm_cap > 0.0
      postconditions:
        - output.recommended_action in ["proceed", "rollback_and_skip", "throttle_lr"]
        - output.recommended_learning_rate > 0.0
      certificate: "Spike is flagged if loss z-score exceeds threshold_std_devs or grad_norm exceeds cap"
      compatible_adapters:
        - ADAPTER-LOSS-SPIKE-WATCHDOG
        - ADAPTER-ROLLBACK-COORDINATOR
      related_algos:
        - ALGO-NN-231
        - ALGO-NN-241
      references:
        - "https://arxiv.org/abs/2204.02311"
        - "https://arxiv.org/abs/2304.03208"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_loss: float,
        current_grad_norm: float,
        loss_history: List[float],
        threshold_std_devs: float = 3.0,
        learning_rate: float = 0.0001,
        grad_norm_cap: float = 10.0,
    ) -> Dict[str, Any]:
        if threshold_std_devs <= 0.0 or learning_rate <= 0.0 or grad_norm_cap <= 0.0:
            raise ValueError("Precondition failed: parameters must be positive")

        n_hist = len(loss_history)
        if n_hist < 5:
            # Insufficient history for reliable statistical z-score
            mean_loss = current_loss
            std_loss = 0.0
            z_score = 0.0
            is_spike = not math.isfinite(current_loss) or current_grad_norm > grad_norm_cap
        else:
            mean_loss = sum(loss_history) / float(n_hist)
            sq_diffs = sum((x - mean_loss) ** 2 for x in loss_history)
            std_loss = math.sqrt(sq_diffs / float(n_hist))
            std_eff = max(1e-6, std_loss)

            if not math.isfinite(current_loss):
                z_score = 999.0
                is_spike = True
            else:
                z_score = (current_loss - mean_loss) / std_eff
                is_spike = (z_score >= threshold_std_devs) or (current_grad_norm > grad_norm_cap)

        if is_spike:
            if not math.isfinite(current_loss) or z_score > 5.0 or current_grad_norm > (grad_norm_cap * 2.0):
                # Catastrophic explosion: rollback and drop offending batch
                action = "rollback_and_skip"
                rec_lr = learning_rate * 0.5
            else:
                # Moderate instability: throttle learning rate
                action = "throttle_lr"
                rec_lr = learning_rate * 0.8
        else:
            action = "proceed"
            rec_lr = learning_rate

        return {
            "is_spike_detected": is_spike,
            "recommended_action": action,
            "recommended_learning_rate": rec_lr,
            "z_score": z_score,
            "rolling_mean": mean_loss,
            "rolling_std": std_loss,
        }
