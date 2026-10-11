from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLongContextTraining:
    """
    ---
    contract:
      algo_id: ALGO-NN-229
      name: NnAlgoLongContextTraining
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - long_context
        - progressive_sequence_length
        - rope_scaling
      inputs:
        type: object
        required:
          - current_step
          - total_steps
          - stage_boundaries
          - stage_sequence_lengths
        properties:
          current_step:
            type: integer
            description: Current global optimization step index.
          total_steps:
            type: integer
            description: Total planned training steps across all stages.
          stage_boundaries:
            type: array
            items:
              type: integer
            description: Step boundaries marking transitions between length stages.
          stage_sequence_lengths:
            type: array
            items:
              type: integer
            description: Active sequence length for each progressive training stage.
          base_rope_theta:
            type: number
            default: 10000.0
            description: Standard base frequency theta for rotary positional embeddings.
          target_rope_theta:
            type: number
            default: 500000.0
            description: Scaled base frequency theta for maximum context length.
      outputs:
        type: object
        required:
          - active_sequence_length
          - active_rope_theta
          - current_stage
          - stage_progress_fraction
          - relative_compute_scale
        properties:
          active_sequence_length:
            type: integer
            description: Sequence token budget for current training step.
          active_rope_theta:
            type: number
            description: Interpolated RoPE base frequency.
          current_stage:
            type: integer
            description: Active curriculum stage index (0-indexed).
          stage_progress_fraction:
            type: number
            description: Fractional progress within current stage in [0.0, 1.0].
          relative_compute_scale:
            type: number
            description: Attention quadratic compute scaling factor (active_len / base_len)^2.
      complexity:
        time: O(len(stage_boundaries))
        space: O(1)
      preconditions:
        - input.current_step >= 0
        - input.total_steps > 0
        - len(input.stage_boundaries) + 1 == len(input.stage_sequence_lengths)
        - all(input.stage_boundaries[i] < input.stage_boundaries[i+1] for i in range(len(input.stage_boundaries)-1))
        - all(l > 0 for l in input.stage_sequence_lengths)
        - input.base_rope_theta > 0.0
      postconditions:
        - output.active_sequence_length > 0
        - output.active_rope_theta >= input.base_rope_theta
        - 0 <= output.current_stage < len(input.stage_sequence_lengths)
        - 0.0 <= output.stage_progress_fraction <= 1.0
      certificate: "Active sequence length correctly transitions across defined stage boundaries"
      compatible_adapters:
        - ADAPTER-PROGRESSIVE-CONTEXT-MANAGER
        - ADAPTER-ROPE-FREQUENCY-SCALER
      related_algos:
        - ALGO-NN-230
        - ALGO-NN-237
      references:
        - "https://arxiv.org/abs/2309.16039"
        - "https://arxiv.org/abs/2310.05202"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_step: int,
        total_steps: int,
        stage_boundaries: List[int],
        stage_sequence_lengths: List[int],
        base_rope_theta: float = 10000.0,
        target_rope_theta: float = 500000.0,
    ) -> Dict[str, Any]:
        num_stages = len(stage_sequence_lengths)
        if current_step < 0 or total_steps <= 0:
            raise ValueError("Precondition failed: steps must be valid")
        if len(stage_boundaries) + 1 != num_stages:
            raise ValueError("Precondition failed: stage_boundaries must have length num_stages - 1")
        if any(stage_boundaries[i] >= stage_boundaries[i + 1] for i in range(len(stage_boundaries) - 1)):
            raise ValueError("Precondition failed: boundaries must be strictly increasing")
        if any(l <= 0 for l in stage_sequence_lengths):
            raise ValueError("Precondition failed: sequence lengths must be positive")

        # Determine current stage
        stage_idx = 0
        for b in stage_boundaries:
            if current_step >= b:
                stage_idx += 1
            else:
                break
        stage_idx = min(num_stages - 1, stage_idx)

        # Stage start and end steps
        stage_start = 0 if stage_idx == 0 else stage_boundaries[stage_idx - 1]
        stage_end = stage_boundaries[stage_idx] if stage_idx < len(stage_boundaries) else total_steps
        stage_len = max(1, stage_end - stage_start)
        stage_progress = min(1.0, max(0.0, float(current_step - stage_start) / float(stage_len)))

        active_seq_len = stage_sequence_lengths[stage_idx]
        base_seq_len = stage_sequence_lengths[0]

        # Logarithmic RoPE theta interpolation based on stage index
        if num_stages > 1:
            ratio = float(stage_idx) / float(num_stages - 1)
            log_base = math.log(base_rope_theta)
            log_target = math.log(target_rope_theta)
            active_theta = math.exp(log_base + ratio * (log_target - log_base))
        else:
            active_theta = base_rope_theta

        compute_scale = (float(active_seq_len) / float(base_seq_len)) ** 2

        return {
            "active_sequence_length": active_seq_len,
            "active_rope_theta": active_theta,
            "current_stage": stage_idx,
            "stage_progress_fraction": stage_progress,
            "relative_compute_scale": compute_scale,
        }
