from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoGradientAccumulation:
    """
    ---
    contract:
      algo_id: ALGO-NN-239
      name: NnAlgoGradientAccumulation
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - gradient_accumulation
        - microbatching
        - token_normalization
        - virtual_batch_size
      inputs:
        type: object
        required:
          - accumulated_gradients
          - microbatch_gradients
          - current_microbatch
          - total_accumulation_steps
        properties:
          accumulated_gradients:
            type: array
            items:
              type: number
            description: Running accumulated gradient vector of dimension P.
          microbatch_gradients:
            type: array
            items:
              type: number
            description: Gradients computed from current microbatch of dimension P.
          current_microbatch:
            type: integer
            description: 1-indexed counter of current microbatch step in [1, total_accumulation_steps].
          total_accumulation_steps:
            type: integer
            description: Virtual batch accumulation factor K >= 1.
          microbatch_tokens:
            type: integer
            default: 1
            description: Number of valid non-padding tokens in current microbatch.
          total_batch_tokens:
            type: integer
            default: 1
            description: Cumulative non-padding tokens across the virtual batch.
          normalization_mode:
            type: string
            default: step
            enum: [step, token]
            description: Loss weighting normalization: step divides by K; token weights by token count.
      outputs:
        type: object
        required:
          - updated_accumulated_gradients
          - is_update_step
          - normalized_gradients_for_step
          - accumulated_steps_count
        properties:
          updated_accumulated_gradients:
            type: array
            items:
              type: number
            description: Updated running accumulated gradients.
          is_update_step:
            type: boolean
            description: True if current microbatch is the final accumulation step triggering an optimizer step.
          normalized_gradients_for_step:
            type: array
            items:
              type: number
            description: Normalized gradient vector ready for optimizer update if is_update_step=True, else empty.
          accumulated_steps_count:
            type: integer
            description: Number of microbatches accumulated so far.
      complexity:
        time: O(P)
        space: O(P)
      preconditions:
        - len(input.accumulated_gradients) == len(input.microbatch_gradients)
        - len(input.microbatch_gradients) > 0
        - 1 <= input.current_microbatch <= input.total_accumulation_steps
        - input.total_accumulation_steps >= 1
      postconditions:
        - len(output.updated_accumulated_gradients) == len(input.accumulated_gradients)
        - output.accumulated_steps_count == input.current_microbatch
      certificate: "On step K, normalized gradients equal arithmetic mean or token-weighted average across all microbatches"
      compatible_adapters:
        - ADAPTER-GRAD-ACCUMULATOR
        - ADAPTER-MICROBATCH-STEP-CONTROLLER
      related_algos:
        - ALGO-NN-233
        - ALGO-NN-244
      references:
        - "https://arxiv.org/abs/1811.06965"
        - "https://arxiv.org/abs/2006.15704"
    ---
    """

    @classmethod
    def forward(
        cls,
        accumulated_gradients: List[float],
        microbatch_gradients: List[float],
        current_microbatch: int,
        total_accumulation_steps: int,
        microbatch_tokens: int = 1,
        total_batch_tokens: int = 1,
        normalization_mode: str = "step",
    ) -> Dict[str, Any]:
        p = len(microbatch_gradients)
        if p == 0 or len(accumulated_gradients) != p:
            raise ValueError("Precondition failed: gradient vectors must have matching positive length")
        if not (1 <= current_microbatch <= total_accumulation_steps):
            raise ValueError("Precondition failed: current_microbatch must be in [1, total_accumulation_steps]")

        # Accumulate gradients
        updated_acc: List[float] = []
        if normalization_mode == "token":
            weight = float(microbatch_tokens) / float(max(1, total_batch_tokens))
            for i in range(p):
                updated_acc.append(accumulated_gradients[i] + microbatch_gradients[i] * weight)
        else:
            for i in range(p):
                updated_acc.append(accumulated_gradients[i] + microbatch_gradients[i])

        is_update = (current_microbatch == total_accumulation_steps)

        if is_update:
            if normalization_mode == "step":
                inv_k = 1.0 / float(total_accumulation_steps)
                step_grads = [g * inv_k for g in updated_acc]
            else:
                step_grads = list(updated_acc)
        else:
            step_grads = []

        return {
            "updated_accumulated_gradients": updated_acc,
            "is_update_step": is_update,
            "normalized_gradients_for_step": step_grads,
            "accumulated_steps_count": current_microbatch,
        }
