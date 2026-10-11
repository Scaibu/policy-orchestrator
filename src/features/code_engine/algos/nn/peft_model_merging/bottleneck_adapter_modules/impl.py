from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoBottleneckAdapterModules:
    """
    ---
    contract:
      algo_id: ALGO-NN-221
      name: NnAlgoBottleneckAdapterModules
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - bottleneck_adapters
        - houlsby_adapters
        - pfeiffer_adapters
        - modular_tuning
      inputs:
        type: object
        required:
          - input_vector
          - w_down
          - b_down
          - w_up
          - b_up
        properties:
          input_vector:
            type: array
            items:
              type: number
            description: Hidden layer activation vector x of dimension d.
          w_down:
            type: array
            items:
              type: array
              items:
                type: number
            description: Down-projection weight matrix of shape (m, d) where m << d.
          b_down:
            type: array
            items:
              type: number
            description: Down-projection bias vector of dimension m.
          w_up:
            type: array
            items:
              type: array
              items:
                type: number
            description: Up-projection weight matrix of shape (d, m).
          b_up:
            type: array
            items:
              type: number
            description: Up-projection bias vector of dimension d.
          activation:
            type: string
            default: gelu
            enum: [gelu, relu]
            description: Non-linear bottleneck activation function.
      outputs:
        type: object
        required:
          - output_vector
          - bottleneck_representation
          - adapter_residual_delta
          - parameter_count
        properties:
          output_vector:
            type: array
            items:
              type: number
            description: Adapted representation y = x + (W_up * f(W_down * x + b_down) + b_up) of dimension d.
          bottleneck_representation:
            type: array
            items:
              type: number
            description: Compressed intermediate latent vector z of dimension m.
          adapter_residual_delta:
            type: array
            items:
              type: number
            description: Additive residual update delta of dimension d.
          parameter_count:
            type: integer
            description: Total trainable parameters in the bottleneck module 2*d*m + d + m.
      complexity:
        time: O(d * m)
        space: O(d + m)
      preconditions:
        - len(input.input_vector) > 0
        - len(input.w_down) > 0
        - all(len(row) == len(input.input_vector) for row in input.w_down)
        - len(input.b_down) == len(input.w_down)
        - len(input.w_up) == len(input.input_vector)
        - all(len(row) == len(input.w_down) for row in input.w_up)
        - len(input.b_up) == len(input.input_vector)
      postconditions:
        - len(output.output_vector) == len(input.input_vector)
        - len(output.bottleneck_representation) == len(input.w_down)
        - len(output.adapter_residual_delta) == len(input.input_vector)
      certificate: "If W_up and b_up are initialized to zero, output_vector equals input_vector exactly"
      compatible_adapters:
        - ADAPTER-BOTTLENECK-LAYER
        - ADAPTER-HOULSBY-TRANSFORMER-INSERTION
      related_algos:
        - ALGO-NN-218
        - ALGO-NN-222
        - ALGO-NN-223
      references:
        - "https://arxiv.org/abs/1902.00751"
        - "https://arxiv.org/abs/2005.00341"
    ---
    """

    @classmethod
    def _gelu(cls, x: float) -> float:
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def forward(
        cls,
        input_vector: List[float],
        w_down: List[List[float]],
        b_down: List[float],
        w_up: List[List[float]],
        b_up: List[float],
        activation: str = "gelu",
    ) -> Dict[str, Any]:
        d = len(input_vector)
        m = len(w_down)
        if d == 0 or m == 0:
            raise ValueError("Precondition failed: input_vector and w_down cannot be empty")
        if any(len(row) != d for row in w_down):
            raise ValueError("Precondition failed: w_down shape must be (m, d)")
        if len(b_down) != m:
            raise ValueError("Precondition failed: b_down length must be m")
        if len(w_up) != d or any(len(row) != m for row in w_up):
            raise ValueError("Precondition failed: w_up shape must be (d, m)")
        if len(b_up) != d:
            raise ValueError("Precondition failed: b_up length must be d")

        # 1. Down-projection: z = W_down * x + b_down
        z: List[float] = []
        for i in range(m):
            val = b_down[i] + sum(w_down[i][j] * input_vector[j] for j in range(d))
            # Apply non-linearity
            if activation == "relu":
                act_val = max(0.0, val)
            else:
                act_val = cls._gelu(val)
            z.append(act_val)

        # 2. Up-projection: delta = W_up * z + b_up
        delta: List[float] = []
        for i in range(d):
            val = b_up[i] + sum(w_up[i][j] * z[j] for j in range(m))
            delta.append(val)

        # 3. Residual connection: y = x + delta
        output_vec = [input_vector[i] + delta[i] for i in range(d)]

        total_params = 2 * d * m + d + m

        return {
            "output_vector": output_vec,
            "bottleneck_representation": z,
            "adapter_residual_delta": delta,
            "parameter_count": total_params,
        }
