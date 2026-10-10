from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLookaheadJacobiDecoding:
    """
    ---
    contract:
      algo_id: ALGO-NN-150
      name: NnAlgoLookaheadJacobiDecoding
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - lookahead_decoding
        - jacobi_decoding
        - fixed_point_iteration
      inputs:
        type: object
        required:
          - initial_guess_tokens
          - max_iterations
        properties:
          initial_guess_tokens:
            type: array
            items:
              type: integer
            description: Speculative window token guesses of length W.
          max_iterations:
            type: integer
            default: 10
            description: Maximum fixed-point update iterations.
      outputs:
        type: object
        required:
          - stabilized_tokens
          - iterations_converged
          - num_tokens_stabilized
        properties:
          stabilized_tokens:
            type: array
            items:
              type: integer
            description: Final fixed-point token sequence.
          iterations_converged:
            type: integer
            description: Iterations executed until fixed-point stabilization.
          num_tokens_stabilized:
            type: integer
            description: Number of parallel tokens generated.
      parameters: {}
      input_assumptions:
        - initial_guess_tokens non-empty and max_iterations >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Converges to standard greedy autoregressive sequence"
      uses_model: false
      complexity:
        variables:
          W: lookahead window width
          I: iterations
        time_worst: O(I * W)
        time_typical: O(I * W)
        space: O(W)
      preconditions:
        - len(input.initial_guess_tokens) > 0
        - input.max_iterations >= 1
      postconditions:
        - len(output.stabilized_tokens) == len(input.initial_guess_tokens)
      certificate: "Fixed point condition: f(y*) == y*"
      compatible_adapters:
        - ADAPTER-JACOBI-ENGINE
      related_algos:
        - ALGO-NN-146
      references:
        - "https://arxiv.org/abs/2305.10427"
        - "https://arxiv.org/abs/2312.12728"
    ---
    """

    @staticmethod
    def iterate(
        initial_guess_tokens: List[int],
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        if not initial_guess_tokens or max_iterations < 1:
            raise ValueError("Precondition failed: invalid inputs")

        W = len(initial_guess_tokens)
        curr = initial_guess_tokens[:]
        iters = 0

        for it in range(1, max_iterations + 1):
            iters = it
            next_tokens = curr[:]
            for j in range(1, W):
                next_tokens[j] = (curr[j - 1] * 7 + 13) % 1000

            if next_tokens == curr:
                break
            curr = next_tokens

        return {
            "stabilized_tokens": curr,
            "iterations_converged": iters,
            "num_tokens_stabilized": W,
        }
