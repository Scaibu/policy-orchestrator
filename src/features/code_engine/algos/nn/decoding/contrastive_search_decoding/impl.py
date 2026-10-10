from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoContrastiveSearchDecoding:
    """
    ---
    contract:
      algo_id: ALGO-NN-145
      name: NnAlgoContrastiveSearchDecoding
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - decoding
        - contrastive_search
        - degeneration_prevention
      inputs:
        type: object
        required:
          - candidate_probs
          - candidate_hidden_states
          - past_hidden_states
          - alpha
        properties:
          candidate_probs:
            type: array
            items:
              type: number
            description: Model probability of candidate tokens in top-k.
          candidate_hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hidden state vectors for candidates of shape (k, d).
          past_hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Context hidden state vectors of shape (seq_len, d).
          alpha:
            type: number
            default: 0.6
            description: Degeneration penalty factor in [0, 1].
      outputs:
        type: object
        required:
          - best_candidate_index
          - candidate_scores
        properties:
          best_candidate_index:
            type: integer
            description: Chosen top candidate maximizing contrastive objective.
          candidate_scores:
            type: array
            items:
              type: number
            description: Contrastive score per candidate.
      parameters: {}
      input_assumptions:
        - 0.0 <= alpha <= 1.0
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
          k: number of candidates
          T: past_hidden_states length
          d: hidden dimension
        time_worst: O(k * T * d)
        time_typical: O(k * T * d)
        space: O(k)
      preconditions:
        - len(input.candidate_probs) > 0 and len(input.candidate_probs) == len(input.candidate_hidden_states)
        - 0.0 <= input.alpha <= 1.0
      postconditions:
        - 0 <= output.best_candidate_index < len(input.candidate_probs)
      certificate: "Score balances model confidence against cosine similarity with past tokens"
      compatible_adapters:
        - ADAPTER-CONTRASTIVE-SEARCH
      related_algos:
        - ALGO-NN-141
      references:
        - "https://arxiv.org/abs/2202.06417"
    ---
    """

    @staticmethod
    def select_candidate(
        candidate_probs: List[float],
        candidate_hidden_states: List[List[float]],
        past_hidden_states: List[List[float]],
        alpha: float = 0.6,
    ) -> Dict[str, Any]:
        if not candidate_probs or len(candidate_probs) != len(candidate_hidden_states):
            raise ValueError("Precondition failed: matching candidates required")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Precondition failed: alpha must be in [0, 1]")

        k = len(candidate_probs)
        d = len(candidate_hidden_states[0])
        scores: List[float] = []

        for i in range(k):
            prob = candidate_probs[i]
            c_vec = candidate_hidden_states[i]
            c_norm = math.sqrt(sum(v * v for v in c_vec)) + 1e-12

            max_sim = 0.0
            if past_hidden_states:
                for p_vec in past_hidden_states:
                    dot = sum(c_vec[j] * p_vec[j] for j in range(d))
                    p_norm = math.sqrt(sum(v * v for v in p_vec)) + 1e-12
                    cos_sim = dot / (c_norm * p_norm)
                    max_sim = max(max_sim, cos_sim)

            score = (1.0 - alpha) * prob - alpha * max_sim
            scores.append(score)

        best_idx = max(range(k), key=lambda i: scores[i])

        return {
            "best_candidate_index": best_idx,
            "candidate_scores": scores,
        }
