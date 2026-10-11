from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMixedPrecisionTraining:
    """
    ---
    contract:
      algo_id: ALGO-NN-231
      name: NnAlgoMixedPrecisionTraining
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - mixed_precision
        - fp16
        - bf16
        - loss_scaling
        - dynamic_scaling
      inputs:
        type: object
        required:
          - raw_gradients
        properties:
          raw_gradients:
            type: array
            items:
              type: number
            description: Gradients computed from scaled backward pass of dimension P.
          loss_scale:
            type: number
            default: 65536.0
            description: Current loss multiplier S > 0 used during backward computation.
          precision_mode:
            type: string
            default: fp16_dynamic
            enum: [fp16_dynamic, bf16, fp32]
            description: Active floating-point arithmetic regime.
          scale_growth_interval:
            type: integer
            default: 2000
            description: Consecutive non-overflow steps required before doubling loss scale.
          consecutive_successful_steps:
            type: integer
            default: 0
            description: Number of consecutive steps completed without overflow.
          fp16_max_representable:
            type: number
            default: 65504.0
            description: Maximum finite value representable in standard IEEE 754 half-precision.
      outputs:
        type: object
        required:
          - unscaled_gradients
          - has_overflow
          - updated_loss_scale
          - step_skipped
          - updated_consecutive_success
          - gradient_norm
        properties:
          unscaled_gradients:
            type: array
            items:
              type: number
            description: Unscaled true gradients g / S ready for FP32 master weight update.
          has_overflow:
            type: boolean
            description: True if any gradient component is NaN, Inf, or exceeds FP16 limit.
          updated_loss_scale:
            type: number
            description: Adjusted loss scale for subsequent iteration.
          step_skipped:
            type: boolean
            description: True if optimizer update must be skipped due to overflow.
          updated_consecutive_success:
            type: integer
            description: Updated tally of consecutive non-overflow iterations.
          gradient_norm:
            type: number
            description: L2 Euclidean norm of the unscaled gradient vector.
      complexity:
        time: O(P)
        space: O(P)
      preconditions:
        - len(input.raw_gradients) > 0
        - input.loss_scale > 0.0
        - input.scale_growth_interval > 0
        - input.consecutive_successful_steps >= 0
      postconditions:
        - len(output.unscaled_gradients) == len(input.raw_gradients)
        - output.updated_loss_scale > 0.0
        - output.gradient_norm >= 0.0
      certificate: "If overflow occurs, step is skipped and loss scale is halved; otherwise scale unscaled gradients by 1/S"
      compatible_adapters:
        - ADAPTER-DYNAMIC-LOSS-SCALER
        - ADAPTER-MIXED-PRECISION-GRAD-UNSCALER
      related_algos:
        - ALGO-NN-232
        - ALGO-NN-242
      references:
        - "https://arxiv.org/abs/1710.03740"
        - "https://arxiv.org/abs/1910.07467"
    ---
    """

    @classmethod
    def forward(
        cls,
        raw_gradients: List[float],
        loss_scale: float = 65536.0,
        precision_mode: str = "fp16_dynamic",
        scale_growth_interval: int = 2000,
        consecutive_successful_steps: int = 0,
        fp16_max_representable: float = 65504.0,
    ) -> Dict[str, Any]:
        p = len(raw_gradients)
        if p == 0:
            raise ValueError("Precondition failed: raw_gradients cannot be empty")
        if loss_scale <= 0.0:
            raise ValueError("Precondition failed: loss_scale must be positive")
        if scale_growth_interval <= 0:
            raise ValueError("Precondition failed: scale_growth_interval must be positive")
        if consecutive_successful_steps < 0:
            raise ValueError("Precondition failed: consecutive_successful_steps must be non-negative")

        if precision_mode == "bf16" or precision_mode == "fp32":
            # BF16 has same dynamic range as FP32 (~10^38), no loss scaling required
            unscaled = list(raw_gradients)
            sq_sum = sum(g * g for g in unscaled if math.isfinite(g))
            grad_norm = math.sqrt(sq_sum)
            return {
                "unscaled_gradients": unscaled,
                "has_overflow": False,
                "updated_loss_scale": 1.0,
                "step_skipped": False,
                "updated_consecutive_success": consecutive_successful_steps + 1,
                "gradient_norm": grad_norm,
            }

        # Check for NaN / Inf / FP16 max overflow
        has_overflow = False
        for g in raw_gradients:
            if not math.isfinite(g) or abs(g) > fp16_max_representable:
                has_overflow = True
                break

        if has_overflow:
            # Step skipped, halve scale immediately, reset counter
            new_scale = max(1.0, loss_scale * 0.5)
            new_consecutive = 0
            # Zero out gradients for safety
            unscaled = [0.0] * p
            grad_norm = 0.0
            step_skipped = True
        else:
            # Valid step: unscale by S
            inv_scale = 1.0 / loss_scale
            unscaled = [g * inv_scale for g in raw_gradients]
            sq_sum = sum(g * g for g in unscaled)
            grad_norm = math.sqrt(sq_sum)
            step_skipped = False

            new_consecutive = consecutive_successful_steps + 1
            if new_consecutive >= scale_growth_interval:
                # Double scale, reset interval counter
                new_scale = loss_scale * 2.0
                new_consecutive = 0
            else:
                new_scale = loss_scale

        return {
            "unscaled_gradients": unscaled,
            "has_overflow": has_overflow,
            "updated_loss_scale": new_scale,
            "step_skipped": step_skipped,
            "updated_consecutive_success": new_consecutive,
            "gradient_norm": grad_norm,
        }
