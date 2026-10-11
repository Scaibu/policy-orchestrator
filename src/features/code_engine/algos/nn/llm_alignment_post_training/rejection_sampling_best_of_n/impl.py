from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRejectionSamplingBestOfN:
    """
    ---
    contract:
      algo_id: ALGO-NN-216
      name: NnAlgoRejectionSamplingBestOfN
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - rejection_sampling
        - best_of_n
        - expert_iteration
        - test_time_compute
      inputs:
        type: object
        required:
          - candidate_scores
          - candidate_tokens
        properties:
          candidate_scores:
            type: array
            items:
              type: number
            description: Reward model or verifier scores for each of the N sampled candidate generations.
          candidate_tokens:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Tokenized sequence representations for each of the N candidates.
          rejection_threshold:
            type: number
            default: 0.5
            description: Minimum score threshold required to accept a candidate for fine-tuning.
          top_k:
            type: integer
            default: 1
            description: Maximum number of highest-scoring candidates to consider for acceptance.
          dedup_similarity_threshold:
            type: number
            default: 0.9
            description: Maximum allowed Jaccard token similarity before flagging a duplicate candidate.
      outputs:
        type: object
        required:
          - best_candidate_index
          - accepted_candidate_indices
          - max_score
          - acceptance_rate
          - diversity_filtered_count
        properties:
          best_candidate_index:
            type: integer
            description: Index of candidate maximizing score (Best-of-N inference selection).
          accepted_candidate_indices:
            type: array
            items:
              type: integer
            description: Indices of accepted candidates meeting score threshold and diversity criteria for RFT.
          max_score:
            type: number
            description: Highest reward score achieved among the N candidates.
          acceptance_rate:
            type: number
            description: Proportion of candidates accepted into training dataset.
          diversity_filtered_count:
            type: integer
            description: Number of high-scoring candidates rejected due to semantic duplication.
      complexity:
        time: O(N log N + K^2 * L)
        space: O(N * L)
      preconditions:
        - len(input.candidate_scores) > 0
        - len(input.candidate_tokens) == len(input.candidate_scores)
        - all(len(seq) > 0 for seq in input.candidate_tokens)
        - input.top_k >= 1
        - 0.0 < input.dedup_similarity_threshold <= 1.0
      postconditions:
        - 0 <= output.best_candidate_index < len(input.candidate_scores)
        - output.max_score == max(input.candidate_scores)
        - 0.0 <= output.acceptance_rate <= 1.0
      certificate: "If top_k == 1 and max_score >= rejection_threshold, accepted_candidate_indices contains exactly best_candidate_index"
      compatible_adapters:
        - ADAPTER-BEST-OF-N-SAMPLER
        - ADAPTER-EXPERT-ITERATION-HARVESTER
      related_algos:
        - ALGO-NN-208
        - ALGO-NN-214
        - ALGO-NN-217
      references:
        - "https://arxiv.org/abs/2308.01825"
        - "https://arxiv.org/abs/2203.02155"
    ---
    """

    @classmethod
    def forward(
        cls,
        candidate_scores: List[float],
        candidate_tokens: List[List[int]],
        rejection_threshold: float = 0.5,
        top_k: int = 1,
        dedup_similarity_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        N = len(candidate_scores)
        if N == 0:
            raise ValueError("Precondition failed: candidate_scores must not be empty")
        if len(candidate_tokens) != N:
            raise ValueError("Precondition failed: candidate_tokens length must match candidate_scores")
        if any(len(seq) == 0 for seq in candidate_tokens):
            raise ValueError("Precondition failed: candidate tokens cannot be empty")
        if top_k < 1:
            raise ValueError("Precondition failed: top_k must be at least 1")
        if not (0.0 < dedup_similarity_threshold <= 1.0):
            raise ValueError("Precondition failed: dedup_similarity_threshold must be in (0, 1]")

        best_idx = max(range(N), key=lambda i: candidate_scores[i])
        max_score = candidate_scores[best_idx]

        # Sort candidates descending by score
        ranked_indices = sorted(range(N), key=lambda i: candidate_scores[i], reverse=True)

        accepted_indices: List[int] = []
        accepted_token_sets: List[set] = []
        dedup_count = 0

        for idx in ranked_indices:
            if len(accepted_indices) >= top_k:
                break
            score = candidate_scores[idx]
            if score < rejection_threshold:
                continue

            current_set = set(candidate_tokens[idx])
            # Check Jaccard overlap against already accepted candidates
            is_dup = False
            for prev_set in accepted_token_sets:
                intersection = len(current_set & prev_set)
                union = len(current_set | prev_set)
                jaccard = float(intersection) / float(union) if union > 0 else 1.0
                if jaccard >= dedup_similarity_threshold:
                    is_dup = True
                    break

            if is_dup:
                dedup_count += 1
            else:
                accepted_indices.append(idx)
                accepted_token_sets.append(current_set)

        acceptance_rate = len(accepted_indices) / float(N)

        return {
            "best_candidate_index": best_idx,
            "accepted_candidate_indices": accepted_indices,
            "max_score": max_score,
            "acceptance_rate": acceptance_rate,
            "diversity_filtered_count": dedup_count,
        }
