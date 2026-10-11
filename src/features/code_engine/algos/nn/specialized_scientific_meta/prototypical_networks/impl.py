from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPrototypicalNetworks:
    """
    ---
    contract:
      algo_id: ALGO-NN-177
      name: NnAlgoPrototypicalNetworks
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - few_shot_learning
        - metric_learning
        - prototypical_networks
        - episodic_classification
      inputs:
        type: object
        required:
          - support_embeddings
          - support_labels
          - query_embeddings
          - num_classes
        properties:
          support_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Embeddings of support examples of shape (N_support, D).
          support_labels:
            type: array
            items:
              type: integer
            description: Class label indices of support examples of length N_support in [0, num_classes - 1].
          query_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Embeddings of query examples of shape (N_query, D).
          num_classes:
            type: integer
            minimum: 1
            description: Total number of classes K.
      outputs:
        type: object
        required:
          - class_prototypes
          - query_log_probabilities
          - predicted_classes
          - min_prototype_distance
        properties:
          class_prototypes:
            type: array
            items:
              type: array
              items:
                type: number
            description: Mean prototype vector for each class of shape (K, D).
          query_log_probabilities:
            type: array
            items:
              type: array
              items:
                type: number
            description: Log softmax distribution over classes for each query of shape (N_query, K).
          predicted_classes:
            type: array
            items:
              type: integer
            description: Argmax predicted class index for each query.
          min_prototype_distance:
            type: number
            description: Smallest pairwise distance between any query and its assigned prototype.
      parameters: {}
      input_assumptions:
        - support_embeddings and query_embeddings have matching uniform dimension D >= 1
        - len(support_embeddings) == len(support_labels) >= num_classes >= 1
        - Each class in [0, num_classes - 1] appears at least once in support_labels
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized log-sum-exp distance softmax"
      uses_model: false
      complexity:
        variables:
          N_s: number of support samples
          N_q: number of query samples
          K: number of classes
          D: embedding dimension
        time_worst: O(N_s * D + N_q * K * D)
        time_typical: O(N_s * D + N_q * K * D)
        space: O(K * D + N_q * K)
      preconditions:
        - len(input.support_embeddings) > 0 and len(input.support_embeddings[0]) > 0
        - len(input.support_embeddings) == len(input.support_labels)
        - len(input.query_embeddings) > 0 and len(input.query_embeddings[0]) == len(input.support_embeddings[0])
        - input.num_classes >= 1
      postconditions:
        - len(output.class_prototypes) == input.num_classes
        - len(output.predicted_classes) == len(input.query_embeddings)
        - output.min_prototype_distance >= 0.0
      certificate: "Class prototypes strictly equal arithmetic means of support class clusters"
      compatible_adapters:
        - ADAPTER-FEW-SHOT-PROTOTYPE-CLASSIFIER
      related_algos:
        - ALGO-NN-178
      references:
        - "https://arxiv.org/abs/1703.05175"
    ---
    """

    @classmethod
    def forward(
        cls,
        support_embeddings: List[List[float]],
        support_labels: List[int],
        query_embeddings: List[List[float]],
        num_classes: int,
    ) -> Dict[str, Any]:
        N_s = len(support_embeddings)
        if N_s == 0 or len(support_embeddings[0]) == 0:
            raise ValueError("Precondition failed: support_embeddings cannot be empty")
        D = len(support_embeddings[0])

        if len(support_labels) != N_s:
            raise ValueError("Precondition failed: support_labels length mismatch")
        if len(query_embeddings) == 0 or any(len(r) != D for r in query_embeddings):
            raise ValueError("Precondition failed: query_embeddings dimension mismatch")
        if num_classes < 1:
            raise ValueError("Precondition failed: num_classes must be >= 1")

        # 1. Compute class prototypes: c_k = (1 / |S_k|) * sum_{x in S_k} x
        counts = [0] * num_classes
        prototypes: List[List[float]] = [[0.0] * D for _ in range(num_classes)]

        for i in range(N_s):
            label = support_labels[i]
            if not (0 <= label < num_classes):
                raise ValueError(f"Precondition failed: invalid label {label} for num_classes {num_classes}")
            counts[label] += 1
            for d in range(D):
                prototypes[label][d] += support_embeddings[i][d]

        for k in range(num_classes):
            if counts[k] == 0:
                raise ValueError(f"Precondition failed: class {k} has zero support examples")
            inv_count = 1.0 / float(counts[k])
            for d in range(D):
                prototypes[k][d] *= inv_count

        # 2. Evaluate query distances and log-softmax probabilities:
        # d(q, c_k) = sum_d (q[d] - c_k[d])^2
        # log p(y = k | q) = -d(q, c_k) - log(sum_j exp(-d(q, c_j)))
        N_q = len(query_embeddings)
        query_log_probs: List[List[float]] = []
        predicted_classes: List[int] = []
        min_dist_found = float("inf")

        for q_idx in range(N_q):
            q_vec = query_embeddings[q_idx]
            neg_dists = []
            for k in range(num_classes):
                dist_sq = 0.0
                for d in range(D):
                    diff = q_vec[d] - prototypes[k][d]
                    dist_sq += diff * diff
                neg_dists.append(-dist_sq)

            max_neg = max(neg_dists)
            lse = max_neg + math.log(sum(math.exp(nd - max_neg) for nd in neg_dists))

            log_probs = [nd - lse for nd in neg_dists]
            best_k = max(range(num_classes), key=lambda k: log_probs[k])

            assigned_dist = -neg_dists[best_k]
            if assigned_dist < min_dist_found:
                min_dist_found = assigned_dist

            query_log_probs.append(log_probs)
            predicted_classes.append(best_k)

        return {
            "class_prototypes": prototypes,
            "query_log_probabilities": query_log_probs,
            "predicted_classes": predicted_classes,
            "min_prototype_distance": min_dist_found if min_dist_found != float("inf") else 0.0,
        }
