from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoRlaifConstitutionCritique:
    """
    ---
    contract:
      algo_id: ALGO-NN-215
      name: NnAlgoRlaifConstitutionCritique
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - rlaif
        - constitutional_ai
        - critique_revision
        - preference_labeling
      inputs:
        type: object
        required:
          - critique_scores
          - principle_weights
        properties:
          critique_scores:
            type: array
            items:
              type: array
              items:
                type: number
            description: Adherence scores in [0, 1] across K constitutional principles for N model outputs.
          principle_weights:
            type: array
            items:
              type: number
            description: Importance weights w_1...w_K for each constitutional rule.
          satisfaction_threshold:
            type: number
            default: 0.8
            description: Minimum score required for a principle to be deemed satisfied.
          human_labels:
            type: array
            items:
              type: integer
            description: Optional ground-truth human binary preference comparisons (1 if A preferred, 0 if B).
          ai_preference_margins:
            type: array
            items:
              type: number
            description: Optional AI judge score differences S(A) - S(B) for agreement verification.
      outputs:
        type: object
        required:
          - composite_constitutional_scores
          - worst_violated_principles
          - all_principles_satisfied
          - ai_human_agreement
        properties:
          composite_constitutional_scores:
            type: array
            items:
              type: number
            description: Weighted constitutional rating score for each candidate output.
          worst_violated_principles:
            type: array
            items:
              type: integer
            description: Index of constitutional principle with lowest adherence for each candidate.
          all_principles_satisfied:
            type: array
            items:
              type: boolean
            description: Boolean flag indicating if every principle met satisfaction_threshold.
          ai_human_agreement:
            type: number
            description: Accuracy of AI synthetic labels against ground-truth human validation samples.
      complexity:
        time: O(N * K)
        space: O(N)
      preconditions:
        - len(input.critique_scores) > 0
        - len(input.principle_weights) > 0
        - all(len(row) == len(input.principle_weights) for row in input.critique_scores)
        - 0.0 < input.satisfaction_threshold <= 1.0
      postconditions:
        - len(output.composite_constitutional_scores) == len(input.critique_scores)
        - len(output.worst_violated_principles) == len(input.critique_scores)
        - len(output.all_principles_satisfied) == len(input.critique_scores)
        - 0.0 <= output.ai_human_agreement <= 1.0
      certificate: "If all critique_scores are 1.0, composite_scores equal sum(principle_weights) and all_principles_satisfied is True"
      compatible_adapters:
        - ADAPTER-RLAIF-EVALUATOR
        - ADAPTER-CONSTITUTIONAL-HARNESS
      related_algos:
        - ALGO-NN-209
        - ALGO-NN-211
        - ALGO-NN-216
      references:
        - "https://arxiv.org/abs/2212.08073"
        - "https://arxiv.org/abs/2309.00267"
    ---
    """

    @classmethod
    def forward(
        cls,
        critique_scores: List[List[float]],
        principle_weights: List[float],
        satisfaction_threshold: float = 0.8,
        human_labels: Optional[List[int]] = None,
        ai_preference_margins: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        N = len(critique_scores)
        if N == 0:
            raise ValueError("Precondition failed: critique_scores must not be empty")
        K = len(principle_weights)
        if K == 0:
            raise ValueError("Precondition failed: principle_weights must not be empty")
        if any(len(row) != K for row in critique_scores):
            raise ValueError("Precondition failed: each critique row must have length K")
        if not (0.0 < satisfaction_threshold <= 1.0):
            raise ValueError("Precondition failed: satisfaction_threshold must be in (0, 1]")

        composite_scores: List[float] = []
        worst_principles: List[int] = []
        all_satisfied: List[bool] = []

        total_weight = sum(principle_weights)
        if total_weight <= 0.0:
            raise ValueError("Precondition failed: sum of principle weights must be positive")

        for i in range(N):
            row = critique_scores[i]
            weighted_sum = sum(row[k] * principle_weights[k] for k in range(K))
            composite_scores.append(weighted_sum)

            # Find worst violated principle (min score)
            worst_k = min(range(K), key=lambda k: row[k])
            worst_principles.append(worst_k)

            # Check if all principles satisfy threshold
            satisfied = all(row[k] >= satisfaction_threshold for k in range(K))
            all_satisfied.append(satisfied)

        # Compute AI-Human agreement rate if validation pairs provided
        if human_labels is not None and ai_preference_margins is not None:
            M = min(len(human_labels), len(ai_preference_margins))
            if M > 0:
                concordant = sum(
                    1 for m in range(M)
                    if (ai_preference_margins[m] > 0.0 and human_labels[m] == 1)
                    or (ai_preference_margins[m] <= 0.0 and human_labels[m] == 0)
                )
                agreement = concordant / float(M)
            else:
                agreement = 1.0
        else:
            agreement = 1.0

        return {
            "composite_constitutional_scores": composite_scores,
            "worst_violated_principles": worst_principles,
            "all_principles_satisfied": all_satisfied,
            "ai_human_agreement": agreement,
        }
