from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoCausalLanguageModeling:
    """
    ---
    contract:
      algo_id: ALGO-NN-103
      name: NnAlgoCausalLanguageModeling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - causal_lm
        - next_token_prediction
        - cross_entropy
      inputs:
        type: object
        required:
          - logits
          - target_tokens
        properties:
          logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Logits matrix of shape (seq_len - 1, vocab_size).
          target_tokens:
            type: array
            items:
              type: integer
            description: Target token indices of shape (seq_len - 1).
          ignore_index:
            type: integer
            default: -100
            description: Target index to ignore in loss computation.
      outputs:
        type: object
        required:
          - loss
          - perplexity
          - token_losses
          - num_active_tokens
        properties:
          loss:
            type: number
            description: Average cross-entropy loss over active tokens.
          perplexity:
            type: number
            description: Perplexity exp(loss).
          token_losses:
            type: array
            items:
              type: number
            description: Per-token loss vector.
          num_active_tokens:
            type: integer
            description: Number of unmasked tokens contributing to loss.
      parameters: {}
      input_assumptions:
        - len(logits) == len(target_tokens)
        - target indices valid or equal to ignore_index
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point bounds"
      uses_model: false
      complexity:
        variables:
          T: sequence length
          V: vocabulary size
        time_worst: O(T * V)
        time_typical: O(T * V)
        space: O(T)
      preconditions:
        - len(input.logits) > 0
        - len(input.logits) == len(input.target_tokens)
      postconditions:
        - output.loss >= 0.0
        - output.perplexity >= 1.0
      certificate: "perplexity == exp(loss)"
      compatible_adapters:
        - ADAPTER-CAUSAL-LM
      related_algos:
        - ALGO-NN-104
        - ALGO-NN-14
      references:
        - "https://doi.org/10.1162/neco.1997.9.8.1735"
    ---
    """

    @staticmethod
    def forward(
        logits: List[List[float]],
        target_tokens: List[int],
        ignore_index: int = -100,
    ) -> Dict[str, Any]:
        if not logits or not target_tokens:
            raise ValueError("Precondition failed: logits and target_tokens must be non-empty")
        if len(logits) != len(target_tokens):
            raise ValueError("Precondition failed: len(logits) == len(target_tokens)")

        V = len(logits[0])
        total_loss = 0.0
        active_count = 0
        token_losses: List[float] = []

        for row, target in zip(logits, target_tokens):
            if target == ignore_index:
                token_losses.append(0.0)
                continue
            if not (0 <= target < V):
                raise ValueError(f"Precondition failed: target token {target} out of range [0, {V})")

            max_l = max(row)
            exp_sum = sum(math.exp(z - max_l) for z in row)
            log_prob = (row[target] - max_l) - math.log(exp_sum)
            nll = -log_prob
            token_losses.append(nll)
            total_loss += nll
            active_count += 1

        avg_loss = (total_loss / active_count) if active_count > 0 else 0.0
        ppl = math.exp(min(avg_loss, 100.0))

        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "token_losses": token_losses,
            "num_active_tokens": active_count,
        }
