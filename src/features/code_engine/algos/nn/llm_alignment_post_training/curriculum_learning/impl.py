from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoCurriculumLearning:
    """
    ---
    contract:
      algo_id: ALGO-NN-207
      name: NnAlgoCurriculumLearning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - curriculum_learning
        - pacing_functions
        - data_scheduling
      inputs:
        type: object
        required:
          - sample_difficulties
          - current_step
          - total_pacing_steps
        properties:
          sample_difficulties:
            type: array
            items:
              type: number
            description: Difficulty metrics (e.g. sequence length, reference model cross-entropy) of length N.
          current_step:
            type: integer
            description: Current training optimization step t >= 0.
          total_pacing_steps:
            type: integer
            description: Total duration T_pace of the curriculum warmup schedule.
          pacing_type:
            type: string
            default: linear
            enum: [linear, root, exponential, step]
            description: Functional form of the pacing schedule g(t).
          initial_fraction:
            type: number
            default: 0.2
            description: Initial proportion of easiest samples available at step 0 in (0, 1].
      outputs:
        type: object
        required:
          - eligible_indices
          - current_difficulty_threshold
          - retention_fraction
          - sorted_sample_indices
        properties:
          eligible_indices:
            type: array
            items:
              type: integer
            description: Indices of training samples whose difficulty is below the active threshold.
          current_difficulty_threshold:
            type: number
            description: Active maximum difficulty score allowed in current training batch.
          retention_fraction:
            type: number
            description: Proportion of total dataset currently eligible for training in [0, 1].
          sorted_sample_indices:
            type: array
            items:
              type: integer
            description: Permutation of sample indices sorted in ascending order of difficulty.
      complexity:
        time: O(N log N)
        space: O(N)
      preconditions:
        - len(input.sample_difficulties) > 0
        - input.current_step >= 0
        - input.total_pacing_steps > 0
        - 0.0 < input.initial_fraction <= 1.0
      postconditions:
        - len(output.eligible_indices) > 0
        - len(output.sorted_sample_indices) == len(input.sample_difficulties)
        - 0.0 < output.retention_fraction <= 1.0
      certificate: "If current_step >= total_pacing_steps, retention_fraction is 1.0 and all samples are eligible"
      compatible_adapters:
        - ADAPTER-CURRICULUM-DATA-LOADER
        - ADAPTER-LLM-PRETRAINING-SCHEDULER
      related_algos:
        - ALGO-NN-208
        - ALGO-NN-229
      references:
        - "https://dl.acm.org/doi/10.1145/1553374.1553380"
    ---
    """

    @classmethod
    def forward(
        cls,
        sample_difficulties: List[float],
        current_step: int,
        total_pacing_steps: int,
        pacing_type: str = "linear",
        initial_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        N = len(sample_difficulties)
        if N == 0:
            raise ValueError("Precondition failed: sample_difficulties must not be empty")
        if current_step < 0:
            raise ValueError("Precondition failed: current_step must be non-negative")
        if total_pacing_steps <= 0:
            raise ValueError("Precondition failed: total_pacing_steps must be positive")
        if not (0.0 < initial_fraction <= 1.0):
            raise ValueError("Precondition failed: initial_fraction must be in (0, 1]")

        # Compute progress ratio tau in [0, 1]
        progress = min(1.0, current_step / float(total_pacing_steps))

        if progress >= 1.0:
            fraction = 1.0
        elif pacing_type == "linear":
            fraction = initial_fraction + (1.0 - initial_fraction) * progress
        elif pacing_type == "root":
            fraction = initial_fraction + (1.0 - initial_fraction) * math.sqrt(progress)
        elif pacing_type == "exponential":
            fraction = initial_fraction * math.pow(1.0 / initial_fraction, progress)
        elif pacing_type == "step":
            steps_num = 4
            step_idx = int(progress * steps_num)
            fraction = min(1.0, initial_fraction + (1.0 - initial_fraction) * (step_idx / float(steps_num)))
        else:
            fraction = initial_fraction + (1.0 - initial_fraction) * progress

        fraction = max(initial_fraction, min(1.0, fraction))

        # Sort indices by difficulty ascending
        sorted_pairs = sorted(enumerate(sample_difficulties), key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in sorted_pairs]

        eligible_count = max(1, min(N, int(math.ceil(fraction * N))))
        eligible_subset = sorted_indices[:eligible_count]
        threshold = sorted_pairs[eligible_count - 1][1]
        retention = float(eligible_count) / float(N)

        return {
            "eligible_indices": eligible_subset,
            "current_difficulty_threshold": threshold,
            "retention_fraction": retention,
            "sorted_sample_indices": sorted_indices,
        }
