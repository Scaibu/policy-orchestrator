from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSyntheticDataGenerationFiltering:
    """
    ---
    contract:
      algo_id: ALGO-NN-245
      name: NnAlgoSyntheticDataGenerationFiltering
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - synthetic_data
        - quality_filtering
        - self_instruct
        - deduplication
        - length_pruning
      inputs:
        type: object
        required:
          - candidate_samples
        properties:
          candidate_samples:
            type: array
            items:
              type: object
            description: Generated candidate records containing text, token_count, and quality_score.
          min_quality_score:
            type: number
            default: 0.7
            description: Minimum verifier/judge quality score threshold in [0.0, 1.0].
          min_token_length:
            type: integer
            default: 10
            description: Minimum acceptable token count.
          max_token_length:
            type: integer
            default: 1000
            description: Maximum acceptable token count.
          jaccard_dedup_threshold:
            type: number
            default: 0.8
            description: N-gram lexical overlap threshold above which candidate is discarded as redundant.
      outputs:
        type: object
        required:
          - accepted_samples
          - rejected_count
          - acceptance_rate
          - rejection_breakdown
          - total_candidates
        properties:
          accepted_samples:
            type: array
            items:
              type: object
            description: High-fidelity filtered synthetic training examples.
          rejected_count:
            type: integer
            description: Total number of discarded candidates.
          acceptance_rate:
            type: number
            description: Ratio of accepted samples to total candidate samples.
          rejection_breakdown:
            type: object
            description: Count of rejections grouped by reason (low_quality, length_violation, duplicate).
          total_candidates:
            type: integer
            description: Total number of evaluated candidates.
      complexity:
        time: O(N * M)
        space: O(N * M)
      preconditions:
        - len(input.candidate_samples) > 0
        - 0.0 <= input.min_quality_score <= 1.0
        - 0 < input.min_token_length <= input.max_token_length
        - 0.0 < input.jaccard_dedup_threshold <= 1.0
      postconditions:
        - len(output.accepted_samples) <= input.total_candidates
        - output.rejected_count == input.total_candidates - len(output.accepted_samples)
        - 0.0 <= output.acceptance_rate <= 1.0
      certificate: "Accepted samples meet quality, length, and uniqueness constraints"
      compatible_adapters:
        - ADAPTER-SYNTHETIC-DATA-FILTER
        - ADAPTER-DEDUP-VERIFIER
      related_algos:
        - ALGO-NN-208
        - ALGO-NN-243
      references:
        - "https://arxiv.org/abs/2212.10560"
        - "https://arxiv.org/abs/2304.12244"
    ---
    """

    @classmethod
    def _words_set(cls, text: str) -> set[str]:
        return set(text.lower().split())

    @classmethod
    def _jaccard(cls, set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return float(intersection) / float(union) if union > 0 else 0.0

    @classmethod
    def forward(
        cls,
        candidate_samples: List[Dict[str, Any]],
        min_quality_score: float = 0.7,
        min_token_length: int = 10,
        max_token_length: int = 1000,
        jaccard_dedup_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        n_candidates = len(candidate_samples)
        if n_candidates == 0:
            raise ValueError("Precondition failed: candidate_samples cannot be empty")
        if not (0.0 <= min_quality_score <= 1.0):
            raise ValueError("Precondition failed: min_quality_score must be in [0.0, 1.0]")
        if min_token_length <= 0 or max_token_length < min_token_length:
            raise ValueError("Precondition failed: invalid token length bounds")

        accepted: List[Dict[str, Any]] = []
        accepted_word_sets: List[set[str]] = []

        rejection_breakdown = {
            "low_quality": 0,
            "length_violation": 0,
            "duplicate": 0,
        }

        for sample in candidate_samples:
            score = sample.get("quality_score", 1.0)
            length = sample.get("token_count", len(sample.get("text", "").split()))
            text = sample.get("text", "")

            # 1. Quality filter
            if score < min_quality_score:
                rejection_breakdown["low_quality"] += 1
                continue

            # 2. Length filter
            if length < min_token_length or length > max_token_length:
                rejection_breakdown["length_violation"] += 1
                continue

            # 3. Deduplication filter against accepted pool
            w_set = cls._words_set(text)
            is_dup = False
            for prev_set in accepted_word_sets:
                if cls._jaccard(w_set, prev_set) >= jaccard_dedup_threshold:
                    is_dup = True
                    break

            if is_dup:
                rejection_breakdown["duplicate"] += 1
                continue

            # Sample passed all filters
            accepted.append(sample)
            accepted_word_sets.append(w_set)

        rejected_count = n_candidates - len(accepted)
        acceptance_rate = float(len(accepted)) / float(n_candidates)

        return {
            "accepted_samples": accepted,
            "rejected_count": rejected_count,
            "acceptance_rate": acceptance_rate,
            "rejection_breakdown": rejection_breakdown,
            "total_candidates": n_candidates,
        }
