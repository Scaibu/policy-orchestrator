from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoMctsAlphazeroMuzero:
    """
    ---
    contract:
      algo_id: ALGO-NN-204
      name: NnAlgoMctsAlphazeroMuzero
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - mcts
        - alphazero
        - muzero
        - puct
        - tree_search
      inputs:
        type: object
        required:
          - prior_probabilities
          - visit_counts
          - q_values
        properties:
          prior_probabilities:
            type: array
            items:
              type: number
            description: Policy prior distribution P(s, a) from neural network across actions.
          visit_counts:
            type: array
            items:
              type: integer
            description: Tree search accumulated visit counts N(s, a) across actions.
          q_values:
            type: array
            items:
              type: number
            description: Mean state-action value estimates Q(s, a) across actions.
          c_puct:
            type: number
            default: 1.25
            description: Base exploration coefficient for PUCT upper confidence bound.
          c_base:
            type: number
            default: 19652.0
            description: Scaling base constant controlling logarithmic PUCT growth.
          temperature:
            type: number
            default: 1.0
            description: Temperature parameter tau for policy distribution extraction.
          dirichlet_noise:
            type: array
            items:
              type: number
            description: Optional exploration noise sampled from Dirichlet distribution.
          dirichlet_epsilon:
            type: number
            default: 0.25
            description: Mixing weight for Dirichlet root noise.
      outputs:
        type: object
        required:
          - selected_action
          - puct_scores
          - ucb_exploration_terms
          - target_policy
          - root_value
        properties:
          selected_action:
            type: integer
            description: Action index maximizing Q(s, a) + U(s, a).
          puct_scores:
            type: array
            items:
              type: number
            description: Combined PUCT scores across actions.
          ucb_exploration_terms:
            type: array
            items:
              type: number
            description: Pure exploration terms U(s, a) across actions.
          target_policy:
            type: array
            items:
              type: number
            description: Normalized visit count target distribution pi(a | s) proportional to N(s, a)^(1/tau).
          root_value:
            type: number
            description: Expected value at root node weighted by visit proportions.
      complexity:
        time: O(A)
        space: O(A)
      preconditions:
        - len(input.prior_probabilities) > 0
        - len(input.visit_counts) == len(input.prior_probabilities)
        - len(input.q_values) == len(input.prior_probabilities)
        - all(n >= 0 for n in input.visit_counts)
        - input.c_puct > 0.0
        - input.temperature > 0.0
      postconditions:
        - 0 <= output.selected_action < len(input.prior_probabilities)
        - len(output.target_policy) == len(input.prior_probabilities)
        - abs(sum(output.target_policy) - 1.0) < 1e-5
      certificate: "If all visit counts are equal and temperature is 1.0, target_policy is uniform 1/A"
      compatible_adapters:
        - ADAPTER-MCTS-PLANNER
        - ADAPTER-ALPHAZERO-AGENT
      related_algos:
        - ALGO-NN-198
        - ALGO-NN-203
      references:
        - "https://arxiv.org/abs/1712.01815"
        - "https://arxiv.org/abs/1911.08265"
    ---
    """

    @classmethod
    def forward(
        cls,
        prior_probabilities: List[float],
        visit_counts: List[int],
        q_values: List[float],
        c_puct: float = 1.25,
        c_base: float = 19652.0,
        temperature: float = 1.0,
        dirichlet_noise: Optional[List[float]] = None,
        dirichlet_epsilon: float = 0.25,
    ) -> Dict[str, Any]:
        A = len(prior_probabilities)
        if A == 0:
            raise ValueError("Precondition failed: prior_probabilities must not be empty")
        if len(visit_counts) != A or len(q_values) != A:
            raise ValueError("Precondition failed: visit_counts and q_values must match action dimension A")
        if any(n < 0 for n in visit_counts):
            raise ValueError("Precondition failed: visit counts must be non-negative")
        if c_puct <= 0.0:
            raise ValueError("Precondition failed: c_puct must be positive")
        if temperature <= 0.0:
            raise ValueError("Precondition failed: temperature must be positive")

        priors = list(prior_probabilities)
        if dirichlet_noise is not None:
            if len(dirichlet_noise) != A:
                raise ValueError("Precondition failed: dirichlet_noise must have length A")
            for a in range(A):
                priors[a] = (1.0 - dirichlet_epsilon) * priors[a] + dirichlet_epsilon * dirichlet_noise[a]

        total_visits = sum(visit_counts)
        sqrt_total = math.sqrt(float(total_visits))

        c_scale = c_puct + math.log((total_visits + c_base + 1.0) / c_base)

        puct_scores: List[float] = []
        u_terms: List[float] = []

        best_score = -float("inf")
        selected_action = 0

        for a in range(A):
            n_a = visit_counts[a]
            u_a = c_scale * priors[a] * (sqrt_total / (1.0 + float(n_a)))
            score_a = q_values[a] + u_a
            puct_scores.append(score_a)
            u_terms.append(u_a)
            if score_a > best_score:
                best_score = score_a
                selected_action = a

        # Target policy extraction pi(a) ~ N(s, a)^(1 / tau)
        if total_visits == 0:
            target_policy = [1.0 / A] * A
            root_val = sum(q_values) / A
        else:
            powered_visits: List[float] = []
            inv_tau = 1.0 / temperature
            for n_a in visit_counts:
                powered_visits.append(math.pow(float(n_a), inv_tau))
            norm = sum(powered_visits)
            if norm == 0.0:
                target_policy = [1.0 / A] * A
            else:
                target_policy = [p / norm for p in powered_visits]

            root_val = sum((visit_counts[a] / total_visits) * q_values[a] for a in range(A))

        return {
            "selected_action": selected_action,
            "puct_scores": puct_scores,
            "ucb_exploration_terms": u_terms,
            "target_policy": target_policy,
            "root_value": root_val,
        }
