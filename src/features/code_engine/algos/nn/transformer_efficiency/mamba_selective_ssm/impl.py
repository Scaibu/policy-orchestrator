from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMambaSelectiveSsm:
    """
    ---
    contract:
      algo_id: ALGO-NN-123
      name: NnAlgoMambaSelectiveSsm
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - mamba
        - selective_ssm
        - state_space_model
      inputs:
        type: object
        required:
          - inputs
          - state_dim
        properties:
          inputs:
            type: array
            items:
              type: number
            description: 1D input sequence x of length L.
          state_dim:
            type: integer
            minimum: 1
            description: Latent SSM state dimension N.
      outputs:
        type: object
        required:
          - outputs
          - final_state
          - delta_history
        properties:
          outputs:
            type: array
            items:
              type: number
            description: Selective SSM filtered outputs of length L.
          final_state:
            type: array
            items:
              type: number
            description: Final hidden state of dimension N.
          delta_history:
            type: array
            items:
              type: number
            description: Input-dependent dynamic delta sequence.
      parameters: {}
      input_assumptions:
        - inputs non-empty and finite
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical precision"
      uses_model: false
      complexity:
        variables:
          L: sequence length
          N: state dimension
        time_worst: O(L * N)
        time_typical: O(L * N)
        space: O(L + N)
      preconditions:
        - len(input.inputs) > 0
        - input.state_dim >= 1
      postconditions:
        - len(output.outputs) == len(input.inputs)
      certificate: "Output sequence matches input length with selective state filtering"
      compatible_adapters:
        - ADAPTER-MAMBA-SSM
      related_algos:
        - ALGO-NN-122
      references:
        - "https://arxiv.org/abs/2312.00752"
    ---
    """

    @staticmethod
    def forward(inputs: List[float], state_dim: int) -> Dict[str, Any]:
        if not inputs or state_dim < 1:
            raise ValueError("Precondition failed: invalid inputs")

        L = len(inputs)
        A = [-1.0 * (i + 1) for i in range(state_dim)]
        h = [0.0] * state_dim
        outputs: List[float] = []
        delta_history: List[float] = []

        for x in inputs:
            delta = math.log(1.0 + math.exp(x * 0.5 + 0.1))
            delta_history.append(delta)

            B_t = [math.tanh(x * 0.1 * (i + 1)) for i in range(state_dim)]
            C_t = [math.cos(x * 0.1 * (i + 1)) for i in range(state_dim)]

            for i in range(state_dim):
                a_bar = math.exp(A[i] * delta)
                b_bar = delta * B_t[i]
                h[i] = a_bar * h[i] + b_bar * x

            y = sum(C_t[i] * h[i] for i in range(state_dim))
            outputs.append(y)

        return {
            "outputs": outputs,
            "final_state": h,
            "delta_history": delta_history,
        }
