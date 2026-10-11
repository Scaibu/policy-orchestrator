from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDeepInterestNetworkAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-194
      name: NnAlgoDeepInterestNetworkAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - recommendation
        - din
        - deep_interest_network
        - target_attention
        - user_behavior_modeling
      inputs:
        type: object
        required:
          - candidate_embedding
          - history_embeddings
          - attn_weights_w1
          - attn_bias_b1
          - attn_weights_w2
          - attn_bias_b2
        properties:
          candidate_embedding:
            type: array
            items:
              type: number
            description: Candidate target item embedding v_A of dimension D.
          history_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Historical sequence of interacted item embeddings of shape (H_history, D).
          attn_weights_w1:
            type: array
            items:
              type: array
              items:
                type: number
            description: First-layer attention MLP weights of shape (D_attn, 4 * D).
          attn_bias_b1:
            type: array
            items:
              type: number
            description: First-layer attention MLP bias of dimension D_attn.
          attn_weights_w2:
            type: array
            items:
              type: number
            description: Second-layer attention scalar projection weights of length D_attn.
          attn_bias_b2:
            type: number
            description: Second-layer attention scalar bias.
          normalize_softmax:
            type: boolean
            default: false
            description: Whether to normalize attention weights via softmax or retain unnormalized activation.
      outputs:
        type: object
        required:
          - user_interest_vector
          - attention_weights
          - total_attention_mass
          - dominant_history_index
        properties:
          user_interest_vector:
            type: array
            items:
              type: number
            description: Candidate-specific activated user interest vector v_U of dimension D.
          attention_weights:
            type: array
            items:
              type: number
            description: Attention coefficients assigned to each historical interaction of length H_history.
          total_attention_mass:
            type: number
            description: Sum of all attention weights across user history.
          dominant_history_index:
            type: integer
            description: Index of historical item with highest attention alignment to the candidate.
      complexity:
        time: O(H_history * (D_attn * D + D))
        space: O(H_history + D + D_attn)
      preconditions:
        - len(input.candidate_embedding) > 0
        - len(input.history_embeddings) > 0 and len(input.history_embeddings[0]) == len(input.candidate_embedding)
        - len(input.attn_weights_w1) > 0 and len(input.attn_weights_w1[0]) == 4 * len(input.candidate_embedding)
        - len(input.attn_bias_b1) == len(input.attn_weights_w1)
        - len(input.attn_weights_w2) == len(input.attn_weights_w1)
      postconditions:
        - len(output.user_interest_vector) == len(input.candidate_embedding)
        - len(output.attention_weights) == len(input.history_embeddings)
        - 0 <= output.dominant_history_index < len(input.history_embeddings)
      certificate: "User interest vector strictly equals weighted sum of historical item embeddings"
      compatible_adapters:
        - ADAPTER-DIN-ATTENTION-EXTRACTOR
        - ADAPTER-CANDIDATE-AWARE-RANKER
      related_algos:
        - ALGO-NN-192
      references:
        - "https://doi.org/10.1145/3219819.3219823"
    ---
    """

    @classmethod
    def forward(
        cls,
        candidate_embedding: List[float],
        history_embeddings: List[List[float]],
        attn_weights_w1: List[List[float]],
        attn_bias_b1: List[float],
        attn_weights_w2: List[float],
        attn_bias_b2: float,
        normalize_softmax: bool = False,
    ) -> Dict[str, Any]:
        D = len(candidate_embedding)
        if D == 0:
            raise ValueError("Precondition failed: candidate_embedding cannot be empty")

        H = len(history_embeddings)
        if H == 0 or any(len(r) != D for r in history_embeddings):
            raise ValueError("Precondition failed: history_embeddings shape mismatch with candidate")

        D_attn = len(attn_bias_b1)
        if len(attn_weights_w1) != D_attn or any(len(r) != 4 * D for r in attn_weights_w1):
            raise ValueError("Precondition failed: attn_weights_w1 must be of shape (D_attn, 4 * D)")
        if len(attn_weights_w2) != D_attn:
            raise ValueError("Precondition failed: attn_weights_w2 dimension mismatch")

        # 1. Evaluate attention score for each historical item e_i with candidate v_A
        raw_scores: List[float] = []

        for i in range(H):
            e_i = history_embeddings[i]
            # Construct interaction vector: [e_i; v_A; e_i * v_A; e_i - v_A] of dimension 4 * D
            inter = list(e_i) + list(candidate_embedding)
            inter += [e_i[d] * candidate_embedding[d] for d in range(D)]
            inter += [e_i[d] - candidate_embedding[d] for d in range(D)]

            # Hidden layer: h = ReLU(W1 * inter + b1)
            h: List[float] = []
            for j in range(D_attn):
                val = sum(attn_weights_w1[j][k] * inter[k] for k in range(4 * D)) + attn_bias_b1[j]
                h.append(max(0.0, val))

            # Output score: a = W2 * h + b2
            score = sum(attn_weights_w2[j] * h[j] for j in range(D_attn)) + attn_bias_b2
            raw_scores.append(score)

        # 2. Normalize weights or apply positive activation
        if normalize_softmax:
            max_s = max(raw_scores)
            exps = [math.exp(s - max_s) for s in raw_scores]
            sum_exp = sum(exps)
            attn_weights = [e / sum_exp for e in exps]
        else:
            # DIN standard uses unbounded / positive activation (e.g. ReLU or Sigmoid)
            attn_weights = [max(0.0, s) for s in raw_scores]

        # 3. Sum weighted history embeddings: v_U = sum_i w_i * e_i
        user_interest = [0.0] * D
        for i in range(H):
            w = attn_weights[i]
            for d in range(D):
                user_interest[d] += w * history_embeddings[i][d]

        total_mass = sum(attn_weights)
        dom_idx = max(range(H), key=lambda idx: attn_weights[idx])

        return {
            "user_interest_vector": user_interest,
            "attention_weights": attn_weights,
            "total_attention_mass": total_mass,
            "dominant_history_index": dom_idx,
        }
