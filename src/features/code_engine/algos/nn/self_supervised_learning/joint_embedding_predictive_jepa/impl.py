from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoJointEmbeddingPredictiveJepa:
    """
    ---
    contract:
      algo_id: ALGO-NN-176
      name: NnAlgoJointEmbeddingPredictiveJepa
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_supervised_learning
        - jepa
        - i_jepa
        - representation_prediction
        - non_generative
      inputs:
        type: object
        required:
          - predicted_target_embeddings
          - actual_target_embeddings
        properties:
          predicted_target_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Predictor network outputs s_hat_y of shape (M_targets, D).
          actual_target_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Target encoder EMA representations s_y of shape (M_targets, D).
          loss_type:
            type: string
            default: l1
            enum:
              - l1
              - l2
            description: Distance loss metric in representation space (l1 or l2).
      outputs:
        type: object
        required:
          - jepa_prediction_loss
          - l1_discrepancy
          - l2_discrepancy
          - mean_cosine_similarity
          - target_embedding_norm
        properties:
          jepa_prediction_loss:
            type: number
            description: Primary normalized JEPA representation prediction loss.
          l1_discrepancy:
            type: number
            description: Mean absolute error in representation space.
          l2_discrepancy:
            type: number
            description: Mean squared error in representation space.
          mean_cosine_similarity:
            type: number
            description: Average cosine similarity between predicted and actual target vectors.
          target_embedding_norm:
            type: number
            description: Average Frobenius norm of the target representations.
      parameters: {}
      input_assumptions:
        - predicted_target_embeddings and actual_target_embeddings have matching shape (M, D) with M >= 1, D >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized cosine normalization with eps 1e-8"
      uses_model: false
      complexity:
        variables:
          M: number of target tokens
          D: embedding dimension
        time_worst: O(M * D)
        time_typical: O(M * D)
        space: O(1)
      preconditions:
        - len(input.predicted_target_embeddings) > 0 and len(input.predicted_target_embeddings[0]) > 0
        - len(input.predicted_target_embeddings) == len(input.actual_target_embeddings)
        - len(input.predicted_target_embeddings[0]) == len(input.actual_target_embeddings[0])
      postconditions:
        - output.jepa_prediction_loss >= 0.0
        - output.l1_discrepancy >= 0.0
        - output.l2_discrepancy >= 0.0
        - -1.0 <= output.mean_cosine_similarity <= 1.000001
      certificate: "JEPA evaluates representation prediction directly in latent space without pixel decoders"
      compatible_adapters:
        - ADAPTER-I-JEPA-TRAINER
        - ADAPTER-V-JEPA-PREDICTOR
      related_algos:
        - ALGO-NN-174
        - ALGO-NN-175
      references:
        - "https://arxiv.org/abs/2301.08243"
    ---
    """

    @classmethod
    def forward(
        cls,
        predicted_target_embeddings: List[List[float]],
        actual_target_embeddings: List[List[float]],
        loss_type: str = "l1",
    ) -> Dict[str, Any]:
        M = len(predicted_target_embeddings)
        if M == 0 or len(predicted_target_embeddings[0]) == 0:
            raise ValueError("Precondition failed: target embeddings cannot be empty")
        D = len(predicted_target_embeddings[0])

        if len(actual_target_embeddings) != M or any(len(r) != D for r in predicted_target_embeddings) or any(len(r) != D for r in actual_target_embeddings):
            raise ValueError("Precondition failed: uniform matrix dimensions required")

        eps = 1e-8
        l1_sum = 0.0
        l2_sum = 0.0
        cos_sum = 0.0
        target_norm_sum = 0.0

        for m in range(M):
            pred_row = predicted_target_embeddings[m]
            target_row = actual_target_embeddings[m]

            dot = 0.0
            norm_p_sq = 0.0
            norm_t_sq = 0.0

            for d in range(D):
                p = pred_row[d]
                t = target_row[d]
                diff = p - t
                l1_sum += abs(diff)
                l2_sum += diff * diff

                dot += p * t
                norm_p_sq += p * p
                norm_t_sq += t * t

            norm_t = math.sqrt(norm_t_sq)
            target_norm_sum += norm_t

            denom = math.sqrt(norm_p_sq * norm_t_sq) + eps
            cos_sum += dot / denom

        total_elements = float(M * D)
        l1_mean = l1_sum / total_elements
        l2_mean = l2_sum / total_elements
        primary_loss = l1_mean if loss_type.lower() == "l1" else l2_mean

        return {
            "jepa_prediction_loss": primary_loss,
            "l1_discrepancy": l1_mean,
            "l2_discrepancy": l2_mean,
            "mean_cosine_similarity": cos_sum / float(M),
            "target_embedding_norm": target_norm_sum / float(M),
        }
