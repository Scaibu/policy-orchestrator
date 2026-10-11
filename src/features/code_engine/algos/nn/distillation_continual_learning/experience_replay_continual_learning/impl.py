from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoExperienceReplayContinualLearning:
    """
    ---
    contract:
      algo_id: ALGO-NN-228
      name: NnAlgoExperienceReplayContinualLearning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - continual_learning
        - experience_replay
        - rehearsal
        - reservoir_sampling
        - catastrophic_forgetting
      inputs:
        type: object
        required:
          - new_task_batch
          - replay_buffer
        properties:
          new_task_batch:
            type: array
            items:
              type: object
            description: Incoming batch of training samples for the current task.
          replay_buffer:
            type: array
            items:
              type: object
            description: Stored historical memory buffer of past task experiences.
          replay_ratio:
            type: number
            default: 0.5
            description: Proportion of replay samples in the blended training batch in [0.0, 1.0].
          max_buffer_capacity:
            type: integer
            default: 100
            description: Maximum capacity of the experience replay buffer.
          sampling_strategy:
            type: string
            default: reservoir
            enum: [reservoir, fifo, balanced]
            description: Memory retention policy when buffer reaches maximum capacity.
          total_seen_samples:
            type: integer
            default: 0
            description: Total cumulative number of samples seen across historical training.
      outputs:
        type: object
        required:
          - blended_batch
          - updated_buffer
          - replay_sample_count
          - new_sample_count
          - buffer_utilization
        properties:
          blended_batch:
            type: array
            items:
              type: object
            description: Combined training batch containing interleaved replay and new task samples.
          updated_buffer:
            type: array
            items:
              type: object
            description: Experience replay buffer updated with new samples.
          replay_sample_count:
            type: integer
            description: Number of historical samples sampled from replay buffer.
          new_sample_count:
            type: integer
            description: Number of new task samples included in blended batch.
          buffer_utilization:
            type: number
            description: Current buffer occupancy ratio len(updated_buffer) / max_buffer_capacity.
      complexity:
        time: O(len(new_task_batch) + len(replay_buffer))
        space: O(max_buffer_capacity + len(blended_batch))
      preconditions:
        - len(input.new_task_batch) > 0
        - input.max_buffer_capacity > 0
        - 0.0 <= input.replay_ratio <= 1.0
        - input.total_seen_samples >= 0
      postconditions:
        - len(output.updated_buffer) <= input.max_buffer_capacity
        - len(output.blended_batch) == output.new_sample_count + output.replay_sample_count
        - output.buffer_utilization <= 1.0
      certificate: "Blended batch contains new_sample_count + replay_sample_count samples, and buffer respects capacity limit"
      compatible_adapters:
        - ADAPTER-REPLAY-MEMORY-BUFFER
        - ADAPTER-REHEARSAL-SAMPLER
      related_algos:
        - ALGO-NN-206
        - ALGO-NN-227
        - ALGO-NN-208
      references:
        - "https://arxiv.org/abs/1902.10486"
        - "https://arxiv.org/abs/2004.10190"
    ---
    """

    @classmethod
    def forward(
        cls,
        new_task_batch: List[Dict[str, Any]],
        replay_buffer: List[Dict[str, Any]],
        replay_ratio: float = 0.5,
        max_buffer_capacity: int = 100,
        sampling_strategy: str = "reservoir",
        total_seen_samples: int = 0,
    ) -> Dict[str, Any]:
        n_new = len(new_task_batch)
        if n_new == 0:
            raise ValueError("Precondition failed: new_task_batch cannot be empty")
        if max_buffer_capacity <= 0:
            raise ValueError("Precondition failed: max_buffer_capacity must be positive")
        if not (0.0 <= replay_ratio <= 1.0):
            raise ValueError("Precondition failed: replay_ratio must be between 0.0 and 1.0")
        if total_seen_samples < 0:
            raise ValueError("Precondition failed: total_seen_samples must be non-negative")

        # 1. Sample replay items if buffer non-empty and ratio > 0
        current_buf_len = len(replay_buffer)
        blended_batch: List[Dict[str, Any]] = []

        if current_buf_len > 0 and replay_ratio > 0.0:
            # Desired replay sample count
            # replay_ratio = k_replay / (k_replay + n_new) => k_replay = n_new * ratio / (1 - ratio)
            if replay_ratio >= 1.0:
                k_replay = current_buf_len
            else:
                k_replay = int(math.ceil(n_new * replay_ratio / (1.0 - replay_ratio)))
            k_replay = max(1, min(current_buf_len, k_replay))

            # Deterministic round-robin / uniform slice of buffer
            step = max(1, current_buf_len // k_replay)
            replayed = [replay_buffer[(i * step) % current_buf_len] for i in range(k_replay)]
            replayed = replayed[:k_replay]
        else:
            replayed = []

        # Interleave replayed and new items
        blended_batch = list(replayed) + list(new_task_batch)

        # 2. Update buffer with new samples
        updated_buffer = list(replay_buffer)

        if sampling_strategy == "fifo":
            # FIFO queue eviction
            for sample in new_task_batch:
                if len(updated_buffer) < max_buffer_capacity:
                    updated_buffer.append(sample)
                else:
                    updated_buffer.pop(0)
                    updated_buffer.append(sample)
        else:
            # Deterministic Reservoir sampling
            current_seen = total_seen_samples
            for sample in new_task_batch:
                current_seen += 1
                if len(updated_buffer) < max_buffer_capacity:
                    updated_buffer.append(sample)
                else:
                    # Deterministic hash replacement based on seen index
                    replace_idx = (current_seen * 2654435761) % current_seen
                    if replace_idx < max_buffer_capacity:
                        updated_buffer[replace_idx] = sample

        utilization = float(len(updated_buffer)) / float(max_buffer_capacity)

        return {
            "blended_batch": blended_batch,
            "updated_buffer": updated_buffer,
            "replay_sample_count": len(replayed),
            "new_sample_count": n_new,
            "buffer_utilization": utilization,
        }
