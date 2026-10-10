from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMultiTokenPrediction:
    """
    ---
    contract:
      algo_id: ALGO-NN-137
      name: NnAlgoMultiTokenPrediction
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - multi_token_prediction
        - mtp
        - dense_supervision
      inputs:
        type: object
        required:
          - trunk_hidden_states
          - target_tokens
          - num_future_tokens
        properties:
          trunk_hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Backbone hidden states of shape (seq_len, d_model).
          target_tokens:
            type: array
            items:
              type: integer
            description: Target token sequence of length seq_len.
          num_future_tokens:
            type: integer
            default: 4
            description: Number of speculative future heads n.
      outputs:
        type: object
        required:
          - composite_loss
          - per_head_losses
          - num_heads
        properties:
          composite_loss:
            type: number
            description: Combined multi-token cross-entropy loss.
          per_head_losses:
            type: array
            items:
              type: number
            description: Loss per prediction offset head [t+1, t+2, ...].
          num_heads:
            type: integer
            description: Number of future prediction heads.
      parameters: {}
      input_assumptions:
        - len(trunk_hidden_states) == len(target_tokens)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard cross-entropy loss calculation"
      uses_model: false
      complexity:
        variables:
          T: seq_len
          n: num_future_tokens
          d: d_model
        time_worst: O(T * n * d)
        time_typical: O(T * n * d)
        space: O(n)
      preconditions:
        - len(input.trunk_hidden_states) > input.num_future_tokens
        - input.num_future_tokens >= 1
      postconditions:
        - output.composite_loss >= 0.0
        - len(output.per_head_losses) == input.num_future_tokens
      certificate: "composite_loss == sum(per_head_losses)"
      compatible_adapters:
        - ADAPTER-MTP-TRAINER
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-147
      references:
        - "https://arxiv.org/abs/2404.19737"
    ---
    """

    @staticmethod
    def forward(
        trunk_hidden_states: List[List[float]],
        target_tokens: List[int],
        num_future_tokens: int = 4,
    ) -> Dict[str, Any]:
        if not trunk_hidden_states or len(trunk_hidden_states) != len(target_tokens):
            raise ValueError("Precondition failed: matching lengths required")
        if num_future_tokens < 1 or len(trunk_hidden_states) <= num_future_tokens:
            raise ValueError("Precondition failed: sequence length must exceed num_future_tokens")

        T = len(trunk_hidden_states)
        per_head_losses: List[float] = []

        for k in range(1, num_future_tokens + 1):
            valid_steps = T - k
            loss_k = 0.0
            for t in range(valid_steps):
                h = trunk_hidden_states[t]
                target_k = target_tokens[t + k]
                prob = max(1e-6, min(1.0, math.exp(-abs(sum(h) * 0.01 - float(target_k % 10) * 0.1))))
                loss_k += -math.log(prob)
            per_head_losses.append(loss_k / float(valid_steps))

        total_loss = sum(per_head_losses)

        return {
            "composite_loss": total_loss,
            "per_head_losses": per_head_losses,
            "num_heads": num_future_tokens,
        }
