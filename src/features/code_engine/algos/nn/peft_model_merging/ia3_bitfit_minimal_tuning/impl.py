from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoIa3BitfitMinimalTuning:
    """
    ---
    contract:
      algo_id: ALGO-NN-223
      name: NnAlgoIa3BitfitMinimalTuning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - ia3
        - bitfit
        - minimal_parameter_tuning
        - activation_rescaling
      inputs:
        type: object
        required:
          - activation_vector
          - ia3_scale_vector
          - trainable_bias
        properties:
          activation_vector:
            type: array
            items:
              type: number
            description: Intermediate hidden activation vector x of dimension d.
          ia3_scale_vector:
            type: array
            items:
              type: number
            description: Trainable multiplicative scaling vector l of dimension d (initialized to 1.0).
          trainable_bias:
            type: array
            items:
              type: number
            description: Trainable additive bias vector b of dimension d.
          mode:
            type: string
            default: ia3
            enum: [ia3, bitfit, combined]
            description: Tuning regime: ia3 (multiplicative l * x), bitfit (additive x + b), or combined.
          base_weight_dimension:
            type: integer
            default: 4096
            description: Base layer matrix dimension (e.g. d x d) for parameter ratio calculation.
      outputs:
        type: object
        required:
          - modulated_output
          - parameter_count
          - parameter_savings_fraction
        properties:
          modulated_output:
            type: array
            items:
              type: number
            description: Adapted activation vector after IA3 scaling and/or BitFit bias addition.
          parameter_count:
            type: integer
            description: Total trainable parameters in this minimal adapter module.
          parameter_savings_fraction:
            type: number
            description: Ratio of trainable parameters to full base matrix parameters.
      complexity:
        time: O(d)
        space: O(d)
      preconditions:
        - len(input.activation_vector) > 0
        - len(input.ia3_scale_vector) == len(input.activation_vector)
        - len(input.trainable_bias) == len(input.activation_vector)
        - input.base_weight_dimension > 0
      postconditions:
        - len(output.modulated_output) == len(input.activation_vector)
        - output.parameter_count > 0
        - 0.0 < output.parameter_savings_fraction < 1.0
      certificate: "If ia3_scale_vector is all 1.0 and trainable_bias is all 0.0, modulated_output matches activation_vector exactly"
      compatible_adapters:
        - ADAPTER-IA3-ACTIVATION-SCALER
        - ADAPTER-BITFIT-BIAS-TUNER
      related_algos:
        - ALGO-NN-218
        - ALGO-NN-221
        - ALGO-NN-222
      references:
        - "https://arxiv.org/abs/2205.05638"
        - "https://arxiv.org/abs/2112.08654"
    ---
    """

    @classmethod
    def forward(
        cls,
        activation_vector: List[float],
        ia3_scale_vector: List[float],
        trainable_bias: List[float],
        mode: str = "ia3",
        base_weight_dimension: int = 4096,
    ) -> Dict[str, Any]:
        d = len(activation_vector)
        if d == 0:
            raise ValueError("Precondition failed: activation_vector must not be empty")
        if len(ia3_scale_vector) != d or len(trainable_bias) != d:
            raise ValueError("Precondition failed: vector dimensions must all equal d")
        if base_weight_dimension <= 0:
            raise ValueError("Precondition failed: base_weight_dimension must be positive")

        out: List[float] = []

        if mode == "ia3":
            # Multiplicative elementwise rescaling: y = l * x
            for i in range(d):
                out.append(ia3_scale_vector[i] * activation_vector[i])
            param_count = d
        elif mode == "bitfit":
            # Additive bias tuning: y = x + b
            for i in range(d):
                out.append(activation_vector[i] + trainable_bias[i])
            param_count = d
        elif mode == "combined":
            # Both: y = l * x + b
            for i in range(d):
                out.append(ia3_scale_vector[i] * activation_vector[i] + trainable_bias[i])
            param_count = 2 * d
        else:
            for i in range(d):
                out.append(ia3_scale_vector[i] * activation_vector[i])
            param_count = d

        base_matrix_params = d * base_weight_dimension
        savings_ratio = float(param_count) / float(base_matrix_params)

        return {
            "modulated_output": out,
            "parameter_count": param_count,
            "parameter_savings_fraction": savings_ratio,
        }
