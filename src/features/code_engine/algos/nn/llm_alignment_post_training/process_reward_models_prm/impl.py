from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoProcessRewardModelsPrm:
    """
    ---
    contract:
      algo_id: ALGO-NN-217
      name: NnAlgoProcessRewardModelsPrm
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - prm
        - process_reward_models
        - step_verification
        - mathematical_reasoning
      inputs:
        type: object
        required:
          - step_predictions
          - step_ground_truth
        properties:
          step_predictions:
            type: array
            items:
              type: number
            description: Model predicted probabilities p_1...p_K in (0, 1) that intermediate reasoning step k is correct.
          step_ground_truth:
            type: array
            items:
              type: integer
            description: Ground truth binary labels y_k in {0, 1} indicating step correctness.
          error_threshold:
            type: number
            default: 0.5
            description: Decision boundary threshold below which a step is flagged as an error.
          aggregation_method:
            type: string
            default: min
            enum: [min, product, mean]
            description: Strategy for aggregating intermediate step scores into full-solution rating.
      outputs:
        type: object
        required:
          - prm_step_loss
          - composite_solution_score
          - first_error_step
          - step_accuracy
          - step_scores
        properties:
          prm_step_loss:
            type: number
            description: Mean step-level binary cross-entropy loss over the reasoning chain.
          composite_solution_score:
            type: number
            description: Overall solution rating computed via weakest-link (min), product, or mean.
          first_error_step:
            type: integer
            description: Index (0-based) of the first step flagged as erroneous (-1 if entire chain is valid).
          step_accuracy:
            type: number
            description: Classification accuracy of PRM predictions against ground-truth step labels.
          step_scores:
            type: array
            items:
              type: number
            description: Array of intermediate step probabilities p_1...p_K.
      complexity:
        time: O(K)
        space: O(K)
      preconditions:
        - len(input.step_predictions) > 0
        - len(input.step_ground_truth) == len(input.step_predictions)
        - all(0.0 <= p <= 1.0 for p in input.step_predictions)
        - all(y in [0, 1] for y in input.step_ground_truth)
        - 0.0 < input.error_threshold < 1.0
      postconditions:
        - output.prm_step_loss >= 0.0
        - 0.0 <= output.composite_solution_score <= 1.0
        - output.first_error_step >= -1
        - 0.0 <= output.step_accuracy <= 1.0
      certificate: "If all step predictions match ground truth exactly, prm_step_loss is near 0.0 and step_accuracy is 1.0"
      compatible_adapters:
        - ADAPTER-PRM-STEP-VERIFIER
        - ADAPTER-MCTS-REASONING-GUIDE
      related_algos:
        - ALGO-NN-204
        - ALGO-NN-209
        - ALGO-NN-214
      references:
        - "https://arxiv.org/abs/2305.20050"
    ---
    """

    @classmethod
    def forward(
        cls,
        step_predictions: List[float],
        step_ground_truth: List[int],
        error_threshold: float = 0.5,
        aggregation_method: str = "min",
    ) -> Dict[str, Any]:
        K = len(step_predictions)
        if K == 0:
            raise ValueError("Precondition failed: step_predictions must not be empty")
        if len(step_ground_truth) != K:
            raise ValueError("Precondition failed: step_ground_truth length must match step_predictions")
        if any(p < 0.0 or p > 1.0 for p in step_predictions):
            raise ValueError("Precondition failed: probabilities must be in [0, 1]")
        if any(y not in (0, 1) for y in step_ground_truth):
            raise ValueError("Precondition failed: ground truth labels must be binary {0, 1}")
        if not (0.0 < error_threshold < 1.0):
            raise ValueError("Precondition failed: error_threshold must be in (0, 1)")

        bce_accum = 0.0
        correct_count = 0
        first_err = -1

        eps = 1e-12

        for k in range(K):
            p = max(eps, min(1.0 - eps, step_predictions[k]))
            y = step_ground_truth[k]

            # Binary cross entropy
            loss_k = - (y * math.log(p) + (1 - y) * math.log(1.0 - p))
            bce_accum += loss_k

            # Classification accuracy against threshold
            pred_binary = 1 if p >= error_threshold else 0
            if pred_binary == y:
                correct_count += 1

            # Detect first error step
            if p < error_threshold and first_err == -1:
                first_err = k

        mean_loss = bce_accum / K
        accuracy = correct_count / float(K)

        # Composite score aggregation
        if aggregation_method == "min":
            composite = min(step_predictions)
        elif aggregation_method == "product":
            prod = 1.0
            for p in step_predictions:
                prod *= p
            composite = prod
        elif aggregation_method == "mean":
            composite = sum(step_predictions) / float(K)
        else:
            composite = min(step_predictions)

        return {
            "prm_step_loss": mean_loss,
            "composite_solution_score": composite,
            "first_error_step": first_err,
            "step_accuracy": accuracy,
            "step_scores": list(step_predictions),
        }
