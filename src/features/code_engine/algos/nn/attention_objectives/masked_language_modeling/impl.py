from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoMaskedLanguageModeling:
    """
    ---
    contract:
      algo_id: ALGO-NN-104
      name: NnAlgoMaskedLanguageModeling
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - masked_language_modeling
        - bert
        - encoder_pretraining
      inputs:
        type: object
        required:
          - logits
          - masked_positions
          - target_tokens
        properties:
          logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoder logits of shape (seq_len, vocab_size).
          masked_positions:
            type: array
            items:
              type: integer
            description: Indices of masked positions in the sequence.
          target_tokens:
            type: array
            items:
              type: integer
            description: Ground truth token indices at the masked positions.
      outputs:
        type: object
        required:
          - loss
          - accuracy
          - predictions
        properties:
          loss:
            type: number
            description: Average cross-entropy loss over masked tokens.
          accuracy:
            type: number
            description: Prediction accuracy on masked positions.
          predictions:
            type: array
            items:
              type: integer
            description: Predicted token indices for each masked position.
      parameters: {}
      input_assumptions:
        - masked_positions and target_tokens have equal length
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical error"
      uses_model: false
      complexity:
        variables:
          M: number of masked tokens
          V: vocab size
        time_worst: O(M * V)
        time_typical: O(M * V)
        space: O(M)
      preconditions:
        - len(input.masked_positions) == len(input.target_tokens)
      postconditions:
        - output.loss >= 0.0
        - 0.0 <= output.accuracy <= 1.0
      certificate: "accuracy is in [0.0, 1.0]"
      compatible_adapters:
        - ADAPTER-BERT-MLM
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-105
      references:
        - "https://arxiv.org/abs/1810.04805"
    ---
    """

    @staticmethod
    def forward(
        logits: List[List[float]],
        masked_positions: List[int],
        target_tokens: List[int],
    ) -> Dict[str, Any]:
        if len(masked_positions) != len(target_tokens):
            raise ValueError("Precondition failed: masked_positions and target_tokens must match length")
        if not masked_positions:
            return {"loss": 0.0, "accuracy": 1.0, "predictions": []}

        V = len(logits[0])
        total_loss = 0.0
        correct = 0
        predictions: List[int] = []

        for pos, target in zip(masked_positions, target_tokens):
            if not (0 <= pos < len(logits)):
                raise ValueError(f"Precondition failed: masked position {pos} out of sequence range")
            if not (0 <= target < V):
                raise ValueError(f"Precondition failed: target token {target} out of range [0, {V})")

            row = logits[pos]
            max_l = max(row)
            exp_sum = sum(math.exp(z - max_l) for z in row)
            log_prob = (row[target] - max_l) - math.log(exp_sum)
            total_loss += -log_prob

            pred = max(range(V), key=lambda idx: row[idx])
            predictions.append(pred)
            if pred == target:
                correct += 1

        M = len(masked_positions)
        return {
            "loss": total_loss / float(M),
            "accuracy": correct / float(M),
            "predictions": predictions,
        }
