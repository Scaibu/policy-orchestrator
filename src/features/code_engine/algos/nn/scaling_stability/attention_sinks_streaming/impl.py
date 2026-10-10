from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoAttentionSinksStreaming:
    """
    ---
    contract:
      algo_id: ALGO-NN-139
      name: NnAlgoAttentionSinksStreaming
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - streaming_llm
        - attention_sinks
        - infinite_context
      inputs:
        type: object
        required:
          - full_token_stream
          - num_sink_tokens
          - window_size
        properties:
          full_token_stream:
            type: array
            items:
              type: integer
            description: Stream of input token IDs.
          num_sink_tokens:
            type: integer
            default: 4
            description: Number of initial anchor sink tokens to pin in cache.
          window_size:
            type: integer
            default: 16
            description: Rolling recent sliding window cache size.
      outputs:
        type: object
        required:
          - retained_cache_tokens
          - total_stream_length
          - evicted_tokens_count
        properties:
          retained_cache_tokens:
            type: array
            items:
              type: integer
            description: Active cached token IDs [sinks + recent window].
          total_stream_length:
            type: integer
            description: Total stream length processed.
          evicted_tokens_count:
            type: integer
            description: Number of middle tokens safely evicted.
      parameters: {}
      input_assumptions:
        - num_sink_tokens >= 1 and window_size >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact buffer partition"
      uses_model: false
      complexity:
        variables:
          T: total stream length
          S: num_sink_tokens
          W: window_size
        time_worst: O(T)
        time_typical: O(T)
        space: O(S + W)
      preconditions:
        - len(input.full_token_stream) > 0
        - input.num_sink_tokens >= 1 and input.window_size >= 1
      postconditions:
        - len(output.retained_cache_tokens) <= input.num_sink_tokens + input.window_size
      certificate: "Retained cache maintains initial sinks and latest recent window"
      compatible_adapters:
        - ADAPTER-STREAMING-LLM
      related_algos:
        - ALGO-NN-117
        - ALGO-NN-119
      references:
        - "https://arxiv.org/abs/2309.17453"
    ---
    """

    @staticmethod
    def filter_stream(
        full_token_stream: List[int],
        num_sink_tokens: int = 4,
        window_size: int = 16,
    ) -> Dict[str, Any]:
        if not full_token_stream or num_sink_tokens < 1 or window_size < 1:
            raise ValueError("Precondition failed: invalid inputs")

        T = len(full_token_stream)
        if T <= num_sink_tokens + window_size:
            return {
                "retained_cache_tokens": full_token_stream[:],
                "total_stream_length": T,
                "evicted_tokens_count": 0,
            }

        sinks = full_token_stream[:num_sink_tokens]
        recent = full_token_stream[-window_size:]
        retained = sinks + recent
        evicted = T - len(retained)

        return {
            "retained_cache_tokens": retained,
            "total_stream_length": T,
            "evicted_tokens_count": evicted,
        }
