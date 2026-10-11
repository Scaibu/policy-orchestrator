from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPipelineParallelism1f1b:
    """
    ---
    contract:
      algo_id: ALGO-NN-236
      name: NnAlgoPipelineParallelism1f1b
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - pipeline_parallelism
        - 1f1b_schedule
        - gpipe
        - pipeline_bubble
        - activation_memory
      inputs:
        type: object
        required:
          - num_pipeline_stages
          - num_microbatches
        properties:
          num_pipeline_stages:
            type: integer
            description: Number of partitioned pipeline stages P.
          num_microbatches:
            type: integer
            description: Total microbatches M in the global batch.
          forward_time_ms:
            type: number
            default: 10.0
            description: Execution time for one microbatch forward pass on a stage (t_f).
          backward_time_ms:
            type: number
            default: 20.0
            description: Execution time for one microbatch backward pass on a stage (t_b).
          schedule_type:
            type: string
            default: 1f1b
            enum: [1f1b, gpipe, interleaved]
            description: Pipeline schedule topology.
      outputs:
        type: object
        required:
          - pipeline_bubble_fraction
          - total_execution_time_ms
          - max_stored_activation_microbatches
          - total_ideal_time_ms
          - idle_bubble_time_ms
        properties:
          pipeline_bubble_fraction:
            type: number
            description: Bubble ratio: fraction of time stages sit idle due to pipeline latency.
          total_execution_time_ms:
            type: number
            description: Total wallclock duration to process all M microbatches.
          max_stored_activation_microbatches:
            type: integer
            description: Peak activation memory footprint measured in microbatches.
          total_ideal_time_ms:
            type: number
            description: Ideal zero-bubble execution time M * (t_f + t_b).
          idle_bubble_time_ms:
            type: number
            description: Total cluster idle time spent in bubbles.
      complexity:
        time: O(1)
        space: O(1)
      preconditions:
        - input.num_pipeline_stages > 0
        - input.num_microbatches >= input.num_pipeline_stages
        - input.forward_time_ms > 0.0
        - input.backward_time_ms > 0.0
      postconditions:
        - 0.0 <= output.pipeline_bubble_fraction < 1.0
        - output.total_execution_time_ms >= output.total_ideal_time_ms
        - output.max_stored_activation_microbatches <= input.num_microbatches
      certificate: "Bubble fraction matches theoretical formula (P - 1) / (M + P - 1) for GPipe / 1F1B"
      compatible_adapters:
        - ADAPTER-PIPELINE-1F1B-SCHEDULER
        - ADAPTER-GPIPE-CONTROLLER
      related_algos:
        - ALGO-NN-234
        - ALGO-NN-235
      references:
        - "https://arxiv.org/abs/1811.06965"
        - "https://arxiv.org/abs/2104.04473"
    ---
    """

    @classmethod
    def forward(
        cls,
        num_pipeline_stages: int,
        num_microbatches: int,
        forward_time_ms: float = 10.0,
        backward_time_ms: float = 20.0,
        schedule_type: str = "1f1b",
    ) -> Dict[str, Any]:
        p = num_pipeline_stages
        m = num_microbatches
        if p <= 0:
            raise ValueError("Precondition failed: num_pipeline_stages must be positive")
        if m < p:
            raise ValueError("Precondition failed: num_microbatches must be at least num_pipeline_stages")
        if forward_time_ms <= 0.0 or backward_time_ms <= 0.0:
            raise ValueError("Precondition failed: forward and backward durations must be positive")

        t_f = forward_time_ms
        t_b = backward_time_ms
        ideal_time = float(m) * (t_f + t_b)

        if schedule_type == "gpipe":
            # GPipe: run all M forward, then all M backward
            # Warmup bubble: (P - 1) * t_f; Cooldown bubble: (P - 1) * t_b
            bubble_time = float(p - 1) * (t_f + t_b)
            total_time = ideal_time + bubble_time
            bubble_fraction = bubble_time / total_time
            # GPipe stores all M activations before backward begins
            max_activations = m
        elif schedule_type == "interleaved":
            # Interleaved 1F1B with virtual stages v = 2: bubble halved
            bubble_time = float(p - 1) * (t_f + t_b) * 0.5
            total_time = ideal_time + bubble_time
            bubble_fraction = bubble_time / total_time
            max_activations = p
        else:
            # Standard 1F1B
            bubble_time = float(p - 1) * (t_f + t_b)
            total_time = ideal_time + bubble_time
            bubble_fraction = bubble_time / total_time
            # 1F1B caps stored activation microbatches to stage count P
            max_activations = p

        return {
            "pipeline_bubble_fraction": bubble_fraction,
            "total_execution_time_ms": total_time,
            "max_stored_activation_microbatches": max_activations,
            "total_ideal_time_ms": ideal_time,
            "idle_bubble_time_ms": bubble_time,
        }
