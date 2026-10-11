from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNonContrastiveSelfSupervision:
    """
    ---
    contract:
      algo_id: ALGO-NN-172
      name: NnAlgoNonContrastiveSelfSupervision
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_supervised_learning
        - non_contrastive
        - byol
        - simsiam
        - collapse_prevention
      inputs:
        type: object
        required:
          - online_pred_view1
          - online_pred_view2
          - target_proj_view1
          - target_proj_view2
        properties:
          online_pred_view1:
            type: array
            items:
              type: number
            description: Online predictor output vector p_1 for augmented view 1 of dimension D.
          online_pred_view2:
            type: array
            items:
              type: number
            description: Online predictor output vector p_2 for augmented view 2 of dimension D.
          target_proj_view1:
            type: array
            items:
              type: number
            description: Target projector representation z_1 for view 1 of dimension D.
          target_proj_view2:
            type: array
            items:
              type: number
            description: Target projector representation z_2 for view 2 of dimension D.
      outputs:
        type: object
        required:
          - symmetric_loss
          - cosine_similarity_12
          - cosine_similarity_21
          - representation_std
        properties:
          symmetric_loss:
            type: number
            description: Symmetrized normalized cosine loss 0.5 * (D(p_1, z_2) + D(p_2, z_1)).
          cosine_similarity_12:
            type: number
            description: Cosine similarity between p_1 and z_2.
          cosine_similarity_21:
            type: number
            description: Cosine similarity between p_2 and z_1.
          representation_std:
            type: number
            description: Empirical standard deviation across features to monitor against collapse.
      parameters: {}
      input_assumptions:
        - All four input vectors have uniform positive length D
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized cosine normalization with eps 1e-8"
      uses_model: false
      complexity:
        variables:
          D: representation vector dimension
        time_worst: O(D)
        time_typical: O(D)
        space: O(1)
      preconditions:
        - len(input.online_pred_view1) > 0
        - len(input.online_pred_view1) == len(input.online_pred_view2) == len(input.target_proj_view1) == len(input.target_proj_view2)
      postconditions:
        - -1.0 <= output.cosine_similarity_12 <= 1.000001
        - -1.0 <= output.cosine_similarity_21 <= 1.000001
        - output.representation_std >= 0.0
      certificate: "Symmetric loss equals 1.0 - 0.5 * (cos_12 + cos_21)"
      compatible_adapters:
        - ADAPTER-BYOL-PRETRAINER
        - ADAPTER-SIMSIAM-TRAINER
      related_algos:
        - ALGO-NN-173
        - ALGO-NN-175
      references:
        - "https://arxiv.org/abs/2006.07733"
        - "https://arxiv.org/abs/2011.10566"
    ---
    """

    @staticmethod
    def _cosine_sim(u: List[float], v: List[float]) -> float:
        eps = 1e-8
        dot = 0.0
        norm_u_sq = 0.0
        norm_v_sq = 0.0
        for a, b in zip(u, v):
            dot += a * b
            norm_u_sq += a * a
            norm_v_sq += b * b
        denom = math.sqrt(norm_u_sq * norm_v_sq) + eps
        return dot / denom

    @classmethod
    def forward(
        cls,
        online_pred_view1: List[float],
        online_pred_view2: List[float],
        target_proj_view1: List[float],
        target_proj_view2: List[float],
    ) -> Dict[str, Any]:
        D = len(online_pred_view1)
        if D == 0:
            raise ValueError("Precondition failed: input vectors cannot be empty")
        if (
            len(online_pred_view2) != D
            or len(target_proj_view1) != D
            or len(target_proj_view2) != D
        ):
            raise ValueError("Precondition failed: vector dimensions must match")

        # 1. Cosine similarity pairs
        cos_12 = cls._cosine_sim(online_pred_view1, target_proj_view2)
        cos_21 = cls._cosine_sim(online_pred_view2, target_proj_view1)

        # 2. Symmetrized negative cosine loss: L = 1 - 0.5 * (cos_12 + cos_21)
        sym_loss = 1.0 - 0.5 * (cos_12 + cos_21)

        # 3. Monitor feature variance / collapse
        mean_z = sum(target_proj_view1) / float(D)
        var_z = sum((x - mean_z) ** 2 for x in target_proj_view1) / float(D)
        std_z = math.sqrt(var_z)

        return {
            "symmetric_loss": sym_loss,
            "cosine_similarity_12": cos_12,
            "cosine_similarity_21": cos_21,
            "representation_std": std_z,
        }
