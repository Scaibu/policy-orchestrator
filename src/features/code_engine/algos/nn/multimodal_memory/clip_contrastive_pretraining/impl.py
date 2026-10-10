from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoClipContrastivePretraining:
    """
    ---
    contract:
      algo_id: ALGO-NN-132
      name: NnAlgoClipContrastivePretraining
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - clip
        - contrastive_learning
        - vision_language
      inputs:
        type: object
        required:
          - image_embeddings
          - text_embeddings
          - logit_scale
        properties:
          image_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Normalized image feature vectors of shape (batch_size, embed_dim).
          text_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Normalized text feature vectors of shape (batch_size, embed_dim).
          logit_scale:
            type: number
            default: 1.0
            description: Learned temperature inverse scale factor exp(tau).
      outputs:
        type: object
        required:
          - loss
          - similarity_matrix
          - accuracy
        properties:
          loss:
            type: number
            description: Symmetric InfoNCE loss across image and text directions.
          similarity_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cosine similarity logit matrix of shape (batch_size, batch_size).
          accuracy:
            type: number
            description: Top-1 diagonal retrieval match accuracy.
      parameters: {}
      input_assumptions:
        - image_embeddings and text_embeddings have matching shape (N, d)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical bounds"
      uses_model: false
      complexity:
        variables:
          N: batch_size
          d: embed_dim
        time_worst: O(N^2 * d)
        time_typical: O(N^2 * d)
        space: O(N^2)
      preconditions:
        - len(input.image_embeddings) > 0
        - len(input.image_embeddings) == len(input.text_embeddings)
        - len(input.image_embeddings[0]) == len(input.text_embeddings[0])
      postconditions:
        - output.loss >= 0.0
        - 0.0 <= output.accuracy <= 1.0
      certificate: "Symmetric cross-entropy formulation over diagonal pairings"
      compatible_adapters:
        - ADAPTER-CLIP-CONTRASTIVE
      related_algos:
        - ALGO-NN-19
      references:
        - "https://arxiv.org/abs/2103.00020"
    ---
    """

    @staticmethod
    def forward(
        image_embeddings: List[List[float]],
        text_embeddings: List[List[float]],
        logit_scale: float = 1.0,
    ) -> Dict[str, Any]:
        if not image_embeddings or len(image_embeddings) != len(text_embeddings):
            raise ValueError("Precondition failed: matching batch size required")

        N = len(image_embeddings)
        d = len(image_embeddings[0])

        sim_matrix: List[List[float]] = []
        for i in range(N):
            row: List[float] = []
            img_v = image_embeddings[i]
            for j in range(N):
                txt_v = text_embeddings[j]
                dot = sum(img_v[k] * txt_v[k] for k in range(d))
                row.append(dot * logit_scale)
            sim_matrix.append(row)

        loss_i2t = 0.0
        correct_i2t = 0
        for i in range(N):
            row = sim_matrix[i]
            max_l = max(row)
            exp_sum = sum(math.exp(z - max_l) for z in row)
            log_prob = (row[i] - max_l) - math.log(exp_sum)
            loss_i2t += -log_prob
            pred_idx = max(range(N), key=lambda idx: row[idx])
            if pred_idx == i:
                correct_i2t += 1

        loss_t2i = 0.0
        for j in range(N):
            col = [sim_matrix[i][j] for i in range(N)]
            max_l = max(col)
            exp_sum = sum(math.exp(z - max_l) for z in col)
            log_prob = (col[j] - max_l) - math.log(exp_sum)
            loss_t2i += -log_prob

        total_loss = (loss_i2t + loss_t2i) / (2.0 * float(N))
        acc = float(correct_i2t) / float(N)

        return {
            "loss": total_loss,
            "similarity_matrix": sim_matrix,
            "accuracy": acc,
        }
