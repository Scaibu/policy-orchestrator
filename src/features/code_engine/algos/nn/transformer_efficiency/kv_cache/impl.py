from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoKvCache:
    """
    ---
    contract:
      algo_id: ALGO-NN-117
      name: NnAlgoKvCache
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - kv_cache
        - autoregressive_decoding
        - inference_engine
      inputs:
        type: object
        required:
          - new_key
          - new_value
        properties:
          new_key:
            type: array
            items:
              type: number
            description: Key vector for current decoding step of shape (d_k).
          new_value:
            type: array
            items:
              type: number
            description: Value vector for current decoding step of shape (d_v).
          cached_keys:
            type: array
            items:
              type: array
              items:
                type: number
            description: Existing cached keys of shape (seq_len, d_k).
          cached_values:
            type: array
            items:
              type: array
              items:
                type: number
            description: Existing cached values of shape (seq_len, d_v).
      outputs:
        type: object
        required:
          - updated_keys
          - updated_values
          - sequence_length
          - memory_bytes
        properties:
          updated_keys:
            type: array
            items:
              type: array
              items:
                type: number
            description: Appended key cache of shape (seq_len + 1, d_k).
          updated_values:
            type: array
            items:
              type: array
              items:
                type: number
            description: Appended value cache of shape (seq_len + 1, d_v).
          sequence_length:
            type: integer
            description: Updated total cached token count.
          memory_bytes:
            type: integer
            description: Approximate memory allocated for cache in bytes (fp16 assumption).
      parameters: {}
      input_assumptions:
        - new_key and new_value are non-empty 1D arrays
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact append operation"
      uses_model: false
      complexity:
        variables:
          T: sequence length
          d: head dimension
        time_worst: O(d)
        time_typical: O(d)
        space: O(T * d)
      preconditions:
        - len(input.new_key) > 0 and len(input.new_value) > 0
      postconditions:
        - output.sequence_length == len(output.updated_keys)
      certificate: "updated_keys[-1] == new_key and updated_values[-1] == new_value"
      compatible_adapters:
        - ADAPTER-KV-CACHE
      related_algos:
        - ALGO-NN-118
      references:
        - "https://arxiv.org/abs/2205.14135"
    ---
    """

    @staticmethod
    def append_step(
        new_key: List[float],
        new_value: List[float],
        cached_keys: Optional[List[List[float]]] = None,
        cached_values: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if not new_key or not new_value:
            raise ValueError("Precondition failed: new_key and new_value must be non-empty")

        keys = [row[:] for row in cached_keys] if cached_keys else []
        values = [row[:] for row in cached_values] if cached_values else []

        keys.append(new_key[:])
        values.append(new_value[:])

        seq_len = len(keys)
        d_k = len(new_key)
        d_v = len(new_value)
        mem_bytes = seq_len * (d_k + d_v) * 2

        return {
            "updated_keys": keys,
            "updated_values": values,
            "sequence_length": seq_len,
            "memory_bytes": mem_bytes,
        }
