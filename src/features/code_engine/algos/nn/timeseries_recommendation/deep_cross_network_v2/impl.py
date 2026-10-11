from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDeepCrossNetworkV2:
    """
    ---
    contract:
      algo_id: ALGO-NN-191
      name: NnAlgoDeepCrossNetworkV2
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - recommendation
        - feature_interactions
        - dcn
        - dcn_v2
        - ctr_prediction
      inputs:
        type: object
        required:
          - input_features_x0
          - cross_weights_w
          - cross_bias_b
          - deep_weights_w
          - deep_bias_b
          - output_weights
        properties:
          input_features_x0:
            type: array
            items:
              type: number
            description: Concatenated dense and sparse categorical embedding vector x_0 of dimension D.
          cross_weights_w:
            type: array
            items:
              type: array
              items:
                type: number
            description: Matrix cross weight W_l of shape (D, D).
          cross_bias_b:
            type: array
            items:
              type: number
            description: Cross layer bias vector b_l of dimension D.
          deep_weights_w:
            type: array
            items:
              type: array
              items:
                type: number
            description: Dense MLP hidden layer weights W_deep of shape (D_deep, D).
          deep_bias_b:
            type: array
            items:
              type: number
            description: Dense MLP hidden layer bias b_deep of dimension D_deep.
          output_weights:
            type: array
            items:
              type: number
            description: Final classification logits vector w_out of dimension D + D_deep.
      outputs:
        type: object
        required:
          - cross_output
          - deep_output
          - predicted_probability
          - cross_layer_energy
        properties:
          cross_output:
            type: array
            items:
              type: number
            description: Explicit feature cross output vector x_1 of dimension D.
          deep_output:
            type: array
            items:
              type: number
            description: Implicit non-linear MLP representation vector of dimension D_deep.
          predicted_probability:
            type: number
            minimum: 0.0
            maximum: 1.0
            description: Sigmoid calibrated CTR / conversion probability in [0, 1].
          cross_layer_energy:
            type: number
            description: L2 norm of the cross feature vector.
      complexity:
        time: O(D^2 + D_deep * D)
        space: O(D + D_deep)
      preconditions:
        - len(input.input_features_x0) > 0
        - len(input.cross_weights_w) == len(input.input_features_x0)
        - len(input.cross_bias_b) == len(input.input_features_x0)
        - len(input.output_weights) == len(input.input_features_x0) + len(input.deep_bias_b)
      postconditions:
        - len(output.cross_output) == len(input.input_features_x0)
        - len(output.deep_output) == len(input.deep_bias_b)
        - 0.0 <= output.predicted_probability <= 1.0
      certificate: "Predicted probability satisfies standard sigmoid output bounded in [0, 1]"
      compatible_adapters:
        - ADAPTER-DCN-V2-CROSS-LAYER
        - ADAPTER-CTR-RANKING-PIPELINE
      related_algos:
        - ALGO-NN-195
      references:
        - "https://arxiv.org/abs/2008.13535"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_features_x0: List[float],
        cross_weights_w: List[List[float]],
        cross_bias_b: List[float],
        deep_weights_w: List[List[float]],
        deep_bias_b: List[float],
        output_weights: List[float],
    ) -> Dict[str, Any]:
        D = len(input_features_x0)
        if D == 0:
            raise ValueError("Precondition failed: input_features_x0 cannot be empty")
        if len(cross_weights_w) != D or any(len(r) != D for r in cross_weights_w):
            raise ValueError("Precondition failed: cross_weights_w must be shape (D, D)")
        if len(cross_bias_b) != D:
            raise ValueError("Precondition failed: cross_bias_b must have length D")

        D_deep = len(deep_bias_b)
        if len(deep_weights_w) != D_deep or any(len(r) != D for r in deep_weights_w):
            raise ValueError("Precondition failed: deep_weights_w must be shape (D_deep, D)")
        if len(output_weights) != D + D_deep:
            raise ValueError("Precondition failed: output_weights must have length D + D_deep")

        # 1. DCN-v2 Matrix Cross Step: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
        # Here x_l = x_0 for a single cross layer
        wx_plus_b: List[float] = []
        for i in range(D):
            val = sum(cross_weights_w[i][j] * input_features_x0[j] for j in range(D)) + cross_bias_b[i]
            wx_plus_b.append(val)

        x_cross: List[float] = []
        cross_energy_sq = 0.0
        for i in range(D):
            # Element-wise product with x_0 plus skip connection
            crossed = input_features_x0[i] * wx_plus_b[i] + input_features_x0[i]
            x_cross.append(crossed)
            cross_energy_sq += crossed * crossed

        # 2. Deep Branch: y_deep = ReLU(W_deep * x_0 + b_deep)
        x_deep: List[float] = []
        for i in range(D_deep):
            val = sum(deep_weights_w[i][j] * input_features_x0[j] for j in range(D)) + deep_bias_b[i]
            x_deep.append(max(0.0, val))

        # 3. Concatenate and project to sigmoid probability: p = sigmoid(w_out^T [x_cross; x_deep])
        stacked = x_cross + x_deep
        logit = sum(output_weights[i] * stacked[i] for i in range(D + D_deep))

        # Stable sigmoid
        if logit >= 0:
            prob = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_z = math.exp(logit)
            prob = exp_z / (1.0 + exp_z)

        return {
            "cross_output": x_cross,
            "deep_output": x_deep,
            "predicted_probability": prob,
            "cross_layer_energy": math.sqrt(cross_energy_sq),
        }
