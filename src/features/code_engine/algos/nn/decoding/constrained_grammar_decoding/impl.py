from __future__ import annotations

import math
from typing import Any, Dict, List, Set


class NnAlgoConstrainedGrammarDecoding:
    """
    ---
    contract:
      algo_id: ALGO-NN-148
      name: NnAlgoConstrainedGrammarDecoding
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - constrained_decoding
        - grammar_masking
        - json_schema
      inputs:
        type: object
        required:
          - logits
          - allowed_tokens
        properties:
          logits:
            type: array
            items:
              type: number
            description: Unconstrained next-token logits of length V.
          allowed_tokens:
            type: array
            items:
              type: integer
            description: Set of valid token IDs permitted by the grammar/schema state.
      outputs:
        type: object
        required:
          - masked_logits
          - selected_token
          - valid_token_count
        properties:
          masked_logits:
            type: array
            items:
              type: number
            description: Logits with disallowed tokens set to -inf.
          selected_token:
            type: integer
            description: Argmax token chosen from the strictly allowed vocabulary subset.
          valid_token_count:
            type: integer
            description: Number of permitted tokens.
      parameters: {}
      input_assumptions:
        - allowed_tokens is non-empty subset of [0, len(logits))
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact masking"
      uses_model: false
      complexity:
        variables:
          V: vocab_size
          A: allowed_tokens count
        time_worst: O(V)
        time_typical: O(V)
        space: O(V)
      preconditions:
        - len(input.logits) > 0
        - len(input.allowed_tokens) > 0
      postconditions:
        - output.selected_token in input.allowed_tokens
      certificate: "Output selected_token is strictly within allowed_tokens"
      compatible_adapters:
        - ADAPTER-GRAMMAR-CONSTRAINT
      related_algos:
        - ALGO-NN-141
      references:
        - "https://arxiv.org/abs/2307.09702"
        - "https://github.com/outlines-dev/outlines"
    ---
    """

    @staticmethod
    def apply_mask(logits: List[float], allowed_tokens: List[int]) -> Dict[str, Any]:
        if not logits or not allowed_tokens:
            raise ValueError("Precondition failed: logits and allowed_tokens must be non-empty")

        V = len(logits)
        allowed_set: Set[int] = set(allowed_tokens)

        masked = [-float("inf")] * V
        for idx in allowed_set:
            if 0 <= idx < V:
                masked[idx] = logits[idx]

        best_token = max(allowed_set, key=lambda idx: logits[idx] if 0 <= idx < V else -float("inf"))

        return {
            "masked_logits": masked,
            "selected_token": best_token,
            "valid_token_count": len(allowed_set),
        }
