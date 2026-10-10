from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class NnAlgoSpanCorruption:
    """
    ---
    contract:
      algo_id: ALGO-NN-105
      name: NnAlgoSpanCorruption
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - span_corruption
        - t5
        - denoising
      inputs:
        type: object
        required:
          - tokens
          - corruption_spans
          - sentinel_start_id
        properties:
          tokens:
            type: array
            items:
              type: integer
            description: Original token sequence.
          corruption_spans:
            type: array
            items:
              type: array
              items:
                type: integer
            description: List of (start_idx, end_idx) non-overlapping token spans to corrupt.
          sentinel_start_id:
            type: integer
            description: Starting token ID for unique sentinel tokens.
      outputs:
        type: object
        required:
          - corrupted_inputs
          - target_sequence
          - num_spans_corrupted
        properties:
          corrupted_inputs:
            type: array
            items:
              type: integer
            description: Input sequence with spans replaced by sentinel tokens.
          target_sequence:
            type: array
            items:
              type: integer
            description: Target decoder sequence containing sentinels followed by missing tokens.
          num_spans_corrupted:
            type: integer
            description: Total count of spans corrupted.
      parameters: {}
      input_assumptions:
        - corruption_spans are sorted and non-overlapping
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact deterministic span replacement"
      uses_model: false
      complexity:
        variables:
          N: sequence length
          S: number of spans
        time_worst: O(N)
        time_typical: O(N)
        space: O(N)
      preconditions:
        - len(input.tokens) > 0
      postconditions:
        - len(output.corrupted_inputs) > 0
        - len(output.target_sequence) > 0
      certificate: "Corrupted inputs contain exactly one sentinel token per corrupted span"
      compatible_adapters:
        - ADAPTER-T5-DENOISING
      related_algos:
        - ALGO-NN-104
      references:
        - "https://arxiv.org/abs/1910.10683"
    ---
    """

    @staticmethod
    def forward(
        tokens: List[int],
        corruption_spans: List[Tuple[int, int]],
        sentinel_start_id: int = 32000,
    ) -> Dict[str, Any]:
        if not tokens:
            raise ValueError("Precondition failed: tokens must be non-empty")

        corrupted_inputs: List[int] = []
        target_seq: List[int] = []

        curr_pos = 0
        sentinel_id = sentinel_start_id

        sorted_spans = sorted(corruption_spans, key=lambda s: s[0])
        for start, end in sorted_spans:
            if start < curr_pos or end > len(tokens) or start >= end:
                raise ValueError(f"Precondition failed: invalid span ({start}, {end})")

            corrupted_inputs.extend(tokens[curr_pos:start])
            corrupted_inputs.append(sentinel_id)

            target_seq.append(sentinel_id)
            target_seq.extend(tokens[start:end])

            curr_pos = end
            sentinel_id += 1

        corrupted_inputs.extend(tokens[curr_pos:])
        target_seq.append(sentinel_id)

        return {
            "corrupted_inputs": corrupted_inputs,
            "target_sequence": target_seq,
            "num_spans_corrupted": len(sorted_spans),
        }
