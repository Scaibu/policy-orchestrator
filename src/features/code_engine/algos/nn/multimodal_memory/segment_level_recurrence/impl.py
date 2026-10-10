from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSegmentLevelRecurrence:
    """
    ---
    contract:
      algo_id: ALGO-NN-134
      name: NnAlgoSegmentLevelRecurrence
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - transformer_xl
        - segment_recurrence
        - long_context
      inputs:
        type: object
        required:
          - current_segment
        properties:
          current_segment:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hidden states of current segment of shape (L_curr, d_model).
          memory_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cached states from previous segment of shape (L_mem, d_model).
      outputs:
        type: object
        required:
          - extended_context
          - updated_memory
          - total_context_length
        properties:
          extended_context:
            type: array
            items:
              type: array
              items:
                type: number
            description: Concatenated [memory, current] states of shape (L_mem + L_curr, d_model).
          updated_memory:
            type: array
            items:
              type: array
              items:
                type: number
            description: Detached memory cache for subsequent segment.
          total_context_length:
            type: integer
            description: Total attended sequence length.
      parameters: {}
      input_assumptions:
        - current_segment is non-empty
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact concatenation"
      uses_model: false
      complexity:
        variables:
          L_curr: current length
          L_mem: memory length
          d: d_model
        time_worst: O((L_curr + L_mem) * d)
        time_typical: O((L_curr + L_mem) * d)
        space: O((L_curr + L_mem) * d)
      preconditions:
        - len(input.current_segment) > 0
      postconditions:
        - len(output.extended_context) >= len(input.current_segment)
      certificate: "total_context_length == len(current_segment) + len(memory_states)"
      compatible_adapters:
        - ADAPTER-TRANSFORMER-XL
      related_algos:
        - ALGO-NN-110
      references:
        - "https://arxiv.org/abs/1901.02860"
    ---
    """

    @staticmethod
    def forward(
        current_segment: List[List[float]],
        memory_states: List[List[float]] | None = None,
    ) -> Dict[str, Any]:
        if not current_segment:
            raise ValueError("Precondition failed: current_segment must be non-empty")

        mem = [row[:] for row in memory_states] if memory_states else []
        extended = mem + [row[:] for row in current_segment]
        updated_mem = [row[:] for row in current_segment]

        return {
            "extended_context": extended,
            "updated_memory": updated_mem,
            "total_context_length": len(extended),
        }
