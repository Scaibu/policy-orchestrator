from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSequentialRecommendationSasrec:
    """
    ---
    contract:
      algo_id: ALGO-NN-192
      name: NnAlgoSequentialRecommendationSasrec
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - recommendation
        - sequential
        - sasrec
        - causal_attention
        - self_attention
      inputs:
        type: object
        required:
          - item_sequence_embeddings
          - position_embeddings
          - candidate_item_embeddings
        properties:
          item_sequence_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Ordered sequence of user interaction item embeddings of shape (T, D).
          position_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Learnable positional embeddings of shape (T, D).
          candidate_item_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Set of candidate item embeddings to score of shape (M_candidates, D).
      outputs:
        type: object
        required:
          - context_sequence
          - candidate_scores
          - causal_attention_matrix
          - top_candidate_index
        properties:
          context_sequence:
            type: array
            items:
              type: array
              items:
                type: number
            description: Causal self-attended sequence representations of shape (T, D).
          candidate_scores:
            type: array
            items:
              type: number
            description: Dot-product ranking scores between the last interaction context and candidate items.
          causal_attention_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Lower-triangular causal attention weights of shape (T, T).
          top_candidate_index:
            type: integer
            description: Index of candidate with highest recommendation score.
      complexity:
        time: O(T^2 * D + M * D)
        space: O(T^2 + T * D + M)
      preconditions:
        - len(input.item_sequence_embeddings) > 0 and len(input.item_sequence_embeddings[0]) > 0
        - len(input.position_embeddings) == len(input.item_sequence_embeddings)
        - len(input.candidate_item_embeddings) > 0
      postconditions:
        - len(output.context_sequence) == len(input.item_sequence_embeddings)
        - len(output.candidate_scores) == len(input.candidate_item_embeddings)
        - 0 <= output.top_candidate_index < len(input.candidate_item_embeddings)
      certificate: "Causal attention matrix is strictly lower triangular with rows summing to 1.0"
      compatible_adapters:
        - ADAPTER-SASREC-SEQUENTIAL-ENCODER
        - ADAPTER-NEXT-ITEM-RETRIEVER
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-194
      references:
        - "https://arxiv.org/abs/1808.09781"
    ---
    """

    @classmethod
    def forward(
        cls,
        item_sequence_embeddings: List[List[float]],
        position_embeddings: List[List[float]],
        candidate_item_embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        T = len(item_sequence_embeddings)
        if T == 0 or len(item_sequence_embeddings[0]) == 0:
            raise ValueError("Precondition failed: item_sequence_embeddings cannot be empty")
        D = len(item_sequence_embeddings[0])

        if len(position_embeddings) != T or any(len(r) != D for r in position_embeddings):
            raise ValueError("Precondition failed: position_embeddings shape mismatch with sequence")
        if len(candidate_item_embeddings) == 0 or any(len(r) != D for r in candidate_item_embeddings):
            raise ValueError("Precondition failed: candidate_item_embeddings dimension mismatch")

        # 1. Sum item and position embeddings: X = E + P
        x_seq: List[List[float]] = []
        for t in range(T):
            row = [item_sequence_embeddings[t][d] + position_embeddings[t][d] for d in range(D)]
            x_seq.append(row)

        scale = 1.0 / math.sqrt(float(D))

        # 2. Compute Causal Self-Attention
        # A[i, j] = (X[i] . X[j]) * scale for j <= i; -inf for j > i
        attn_matrix: List[List[float]] = []
        context_seq: List[List[float]] = []

        for i in range(T):
            scores: List[float] = []
            for j in range(i + 1):
                dot = sum(x_seq[i][d] * x_seq[j][d] for d in range(D))
                scores.append(dot * scale)

            # Softmax over causal history [0, i]
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exps)
            probs = [e / sum_exp for e in exps]

            # Full row of length T with zeros for future positions
            full_row = probs + [0.0] * (T - i - 1)
            attn_matrix.append(full_row)

            # Aggregate context vector: c_i = sum_{j <= i} w_{i, j} * X[j]
            c_i = [0.0] * D
            for j in range(i + 1):
                p = probs[j]
                for d in range(D):
                    c_i[d] += p * x_seq[j][d]
            context_seq.append(c_i)

        # 3. Score candidate items against the latest interaction context (c_{T-1})
        last_context = context_seq[-1]
        candidate_scores: List[float] = []
        for cand in candidate_item_embeddings:
            score = sum(last_context[d] * cand[d] for d in range(D))
            candidate_scores.append(score)

        top_idx = max(range(len(candidate_scores)), key=lambda idx: candidate_scores[idx])

        return {
            "context_sequence": context_seq,
            "candidate_scores": candidate_scores,
            "causal_attention_matrix": attn_matrix,
            "top_candidate_index": top_idx,
        }
