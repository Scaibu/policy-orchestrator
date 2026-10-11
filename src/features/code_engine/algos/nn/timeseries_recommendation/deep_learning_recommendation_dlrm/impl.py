from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDeepLearningRecommendationDlrm:
    """
    ---
    contract:
      algo_id: ALGO-NN-195
      name: NnAlgoDeepLearningRecommendationDlrm
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - recommendation
        - dlrm
        - embedding_interactions
        - bottom_top_mlp
        - ctr_prediction
      inputs:
        type: object
        required:
          - dense_features
          - sparse_embeddings
          - bottom_mlp_weights
          - bottom_mlp_bias
          - top_mlp_weights
          - top_mlp_bias
        properties:
          dense_features:
            type: array
            items:
              type: number
            description: Continuous dense features x_dense of dimension D_dense.
          sparse_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: S categorical feature embedding vectors of shape (S_sparse, D).
          bottom_mlp_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Bottom MLP linear weights mapping dense features to dimension D of shape (D, D_dense).
          bottom_mlp_bias:
            type: array
            items:
              type: number
            description: Bottom MLP bias vector of dimension D.
          top_mlp_weights:
            type: array
            items:
              type: number
            description: Top MLP scoring weights of dimension D + S_sparse * (S_sparse + 1) // 2.
          top_mlp_bias:
            type: number
            description: Top MLP scalar bias.
      outputs:
        type: object
        required:
          - bottom_representation
          - interaction_vector
          - predicted_probability
          - interaction_energy
        properties:
          bottom_representation:
            type: array
            items:
              type: number
            description: Processed dense feature vector v_0 of dimension D.
          interaction_vector:
            type: array
            items:
              type: number
            description: Lower-triangular pairwise dot product interactions of length S*(S+1)/2.
          predicted_probability:
            type: number
            minimum: 0.0
            maximum: 1.0
            description: Calibrated click-through rate probability in [0, 1].
          interaction_energy:
            type: number
            description: L2 norm of the pairwise interaction vector.
      complexity:
        time: O(D * D_dense + S^2 * D + D_top)
        space: O(D + S^2)
      preconditions:
        - len(input.dense_features) > 0
        - len(input.sparse_embeddings) > 0 and len(input.sparse_embeddings[0]) == len(input.bottom_mlp_bias)
        - len(input.bottom_mlp_weights) == len(input.bottom_mlp_bias)
        - len(input.top_mlp_weights) == len(input.bottom_mlp_bias) + (len(input.sparse_embeddings) * (len(input.sparse_embeddings) + 1)) // 2
      postconditions:
        - len(output.bottom_representation) == len(input.bottom_mlp_bias)
        - len(output.interaction_vector) == (len(input.sparse_embeddings) * (len(input.sparse_embeddings) + 1)) // 2
        - 0.0 <= output.predicted_probability <= 1.0
      certificate: "Predicted probability strictly matches sigmoid activation bounded in [0, 1]"
      compatible_adapters:
        - ADAPTER-DLRM-INTERACTION-CORE
        - ADAPTER-LARGE-EMBEDDING-RANKER
      related_algos:
        - ALGO-NN-191
      references:
        - "https://arxiv.org/abs/1906.00091"
    ---
    """

    @classmethod
    def forward(
        cls,
        dense_features: List[float],
        sparse_embeddings: List[List[float]],
        bottom_mlp_weights: List[List[float]],
        bottom_mlp_bias: List[float],
        top_mlp_weights: List[float],
        top_mlp_bias: float,
    ) -> Dict[str, Any]:
        D_dense = len(dense_features)
        if D_dense == 0:
            raise ValueError("Precondition failed: dense_features cannot be empty")

        D = len(bottom_mlp_bias)
        if D == 0 or len(bottom_mlp_weights) != D or any(len(r) != D_dense for r in bottom_mlp_weights):
            raise ValueError("Precondition failed: bottom_mlp_weights dimension mismatch")

        S = len(sparse_embeddings)
        if S == 0 or any(len(r) != D for r in sparse_embeddings):
            raise ValueError("Precondition failed: sparse_embeddings must be shape (S, D)")

        num_interactions = (S * (S + 1)) // 2
        expected_top_dim = D + num_interactions
        if len(top_mlp_weights) != expected_top_dim:
            raise ValueError(
                f"Precondition failed: top_mlp_weights length {len(top_mlp_weights)} != expected {expected_top_dim}"
            )

        # 1. Process dense features through Bottom MLP: v_0 = ReLU(W_bot * x_dense + b_bot)
        v0: List[float] = []
        for d in range(D):
            val = sum(bottom_mlp_weights[d][j] * dense_features[j] for j in range(D_dense)) + bottom_mlp_bias[d]
            v0.append(max(0.0, val))

        # 2. Combine [v0; sparse_embeddings] into V of shape (S + 1, D)
        all_vectors = [v0] + sparse_embeddings  # length S + 1

        # 3. Explicit pairwise dot product interactions: lower triangle of V * V^T
        interactions: List[float] = []
        for i in range(1, S + 1):
            for j in range(i):
                # Dot product between vector i and vector j
                dot = sum(all_vectors[i][d] * all_vectors[j][d] for d in range(D))
                interactions.append(dot)

        # 4. Concatenate v0 and interactions into Top MLP input
        top_input = v0 + interactions
        top_logit = sum(top_mlp_weights[i] * top_input[i] for i in range(expected_top_dim)) + top_mlp_bias

        # 5. Sigmoid calibration
        if top_logit >= 0:
            prob = 1.0 / (1.0 + math.exp(-top_logit))
        else:
            ez = math.exp(top_logit)
            prob = ez / (1.0 + ez)

        inter_energy = math.sqrt(sum(x * x for x in interactions))

        return {
            "bottom_representation": v0,
            "interaction_vector": interactions,
            "predicted_probability": prob,
            "interaction_energy": inter_energy,
        }
