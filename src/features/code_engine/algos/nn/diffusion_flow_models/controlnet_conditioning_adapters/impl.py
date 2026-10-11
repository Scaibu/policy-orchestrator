from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoControlnetConditioningAdapters:
    """
    ---
    contract:
      algo_id: ALGO-NN-170
      name: NnAlgoControlnetConditioningAdapters
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - controlnet
        - zero_convolution
        - conditioning_adapter
        - spatial_guidance
      inputs:
        type: object
        required:
          - feature_x
          - condition_c
          - locked_weights
          - adapter_weights
          - zero_conv_in_weights
          - zero_conv_out_weights
        properties:
          feature_x:
            type: array
            items:
              type: number
            description: Base activation feature vector x of dimension D.
          condition_c:
            type: array
            items:
              type: number
            description: Conditioning signal vector c (e.g. edge/depth) of dimension D.
          locked_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Frozen base network block weight matrix of shape (D, D).
          adapter_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Trainable cloned adapter block weight matrix of shape (D, D).
          zero_conv_in_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Input zero-convolution weights of shape (D, D).
          zero_conv_out_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Output zero-convolution weights of shape (D, D).
          control_scale:
            type: number
            default: 1.0
            description: User conditioning strength multiplier.
      outputs:
        type: object
        required:
          - combined_output
          - base_feature_output
          - adapter_feature_output
          - zero_injection_norm
        properties:
          combined_output:
            type: array
            items:
              type: number
            description: Synthesized output y = F_locked(x) + scale * Z_out(F_adapter(x + Z_in(c))).
          base_feature_output:
            type: array
            items:
              type: number
            description: Output from frozen base block F_locked(x).
          adapter_feature_output:
            type: array
            items:
              type: number
            description: Injected residual signal from adapter pathway.
          zero_injection_norm:
            type: number
            description: L2 norm of the injected adapter signal.
      parameters: {}
      input_assumptions:
        - feature_x and condition_c have uniform dimension D
        - all weight matrices have shape (D, D)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact matrix-vector multiplication"
      uses_model: false
      complexity:
        variables:
          D: feature vector dimension
        time_worst: O(D^2)
        time_typical: O(D^2)
        space: O(D)
      preconditions:
        - len(input.feature_x) > 0 and len(input.feature_x) == len(input.condition_c)
        - len(input.locked_weights) == len(input.feature_x) and len(input.locked_weights[0]) == len(input.feature_x)
        - len(input.adapter_weights) == len(input.feature_x) and len(input.adapter_weights[0]) == len(input.feature_x)
        - len(input.zero_conv_in_weights) == len(input.feature_x) and len(input.zero_conv_in_weights[0]) == len(input.feature_x)
        - len(input.zero_conv_out_weights) == len(input.feature_x) and len(input.zero_conv_out_weights[0]) == len(input.feature_x)
      postconditions:
        - len(output.combined_output) == len(input.feature_x)
        - output.zero_injection_norm >= 0.0
      certificate: "When zero_conv_out_weights is zero, combined_output strictly equals base_feature_output"
      compatible_adapters:
        - ADAPTER-CONTROLNET-PIPELINE
      related_algos:
        - ALGO-NN-161
        - ALGO-NN-166
        - ALGO-NN-167
      references:
        - "https://arxiv.org/abs/2302.05543"
    ---
    """

    @staticmethod
    def _matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
        D = len(vector)
        out = []
        for i in range(D):
            val = 0.0
            for j in range(D):
                val += matrix[i][j] * vector[j]
            out.append(val)
        return out

    @classmethod
    def forward(
        cls,
        feature_x: List[float],
        condition_c: List[float],
        locked_weights: List[List[float]],
        adapter_weights: List[List[float]],
        zero_conv_in_weights: List[List[float]],
        zero_conv_out_weights: List[List[float]],
        control_scale: float = 1.0,
    ) -> Dict[str, Any]:
        D = len(feature_x)
        if D == 0 or len(condition_c) != D:
            raise ValueError("Precondition failed: feature and condition dimension mismatch")

        for name, mat in [
            ("locked_weights", locked_weights),
            ("adapter_weights", adapter_weights),
            ("zero_conv_in_weights", zero_conv_in_weights),
            ("zero_conv_out_weights", zero_conv_out_weights),
        ]:
            if len(mat) != D or any(len(r) != D for r in mat):
                raise ValueError(f"Precondition failed: {name} dimension mismatch")

        # 1. Base model path: y_base = locked_weights * x
        y_base = cls._matvec(locked_weights, feature_x)

        # 2. ControlNet path:
        # a. Zero-conv on condition: c_in = zero_conv_in * c
        c_in = cls._matvec(zero_conv_in_weights, condition_c)

        # b. Add to input: x_cond = x + c_in
        x_cond = [feature_x[i] + c_in[i] for i in range(D)]

        # c. Adapter block: h_adapter = adapter_weights * x_cond
        h_adapter = cls._matvec(adapter_weights, x_cond)

        # d. Zero-conv on adapter output: y_adapter = zero_conv_out * h_adapter
        y_adapter = cls._matvec(zero_conv_out_weights, h_adapter)

        # 3. Combine: y = y_base + control_scale * y_adapter
        combined_output = []
        injection_sq_sum = 0.0
        for i in range(D):
            inj = control_scale * y_adapter[i]
            injection_sq_sum += inj * inj
            combined_output.append(y_base[i] + inj)

        return {
            "combined_output": combined_output,
            "base_feature_output": y_base,
            "adapter_feature_output": y_adapter,
            "zero_injection_norm": math.sqrt(injection_sq_sum),
        }
