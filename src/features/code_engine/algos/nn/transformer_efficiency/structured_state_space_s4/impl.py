from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoStructuredStateSpaceS4:
    """
    ---
    contract:
      algo_id: ALGO-NN-122
      name: NnAlgoStructuredStateSpaceS4
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - state_space_model
        - s4
        - hippo
      inputs:
        type: object
        required:
          - inputs
          - state_dim
          - delta
        properties:
          inputs:
            type: array
            items:
              type: number
            description: 1D input sequence x of length L.
          state_dim:
            type: integer
            minimum: 1
            description: SSM latent state dimension N.
          delta:
            type: number
            minimum: 0.0
            exclusiveMinimum: true
            description: Discretization step size Delta > 0.
      outputs:
        type: object
        required:
          - outputs
          - final_state
          - state_dim
        properties:
          outputs:
            type: array
            items:
              type: number
            description: Convolved output sequence y of length L.
          final_state:
            type: array
            items:
              type: number
            description: Final latent state h_L of dimension N.
          state_dim:
            type: integer
            description: State dimension.
      parameters: {}
      input_assumptions:
        - delta > 0.0 and inputs non-empty
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard Euler/Bilinear discretization error"
      uses_model: false
      complexity:
        variables:
          L: sequence length
          N: state dimension
        time_worst: O(L * N)
        time_typical: O(L * N)
        space: O(N + L)
      preconditions:
        - len(input.inputs) > 0
        - input.state_dim >= 1
        - input.delta > 0.0
      postconditions:
        - len(output.outputs) == len(input.inputs)
        - len(output.final_state) == input.state_dim
      certificate: "Outputs length exactly equals input length"
      compatible_adapters:
        - ADAPTER-S4-SSM
      related_algos:
        - ALGO-NN-123
      references:
        - "https://arxiv.org/abs/2111.00396"
    ---
    """

    @staticmethod
    def forward(inputs: List[float], state_dim: int, delta: float = 0.01) -> Dict[str, Any]:
        if not inputs or state_dim < 1 or delta <= 0.0:
            raise ValueError("Precondition failed: invalid inputs or dimensions")

        A = [-1.0 / (float(i) + 1.0) for i in range(state_dim)]
        B = [1.0] * state_dim
        C = [1.0 / math.sqrt(state_dim)] * state_dim

        A_bar = [math.exp(a * delta) for a in A]
        B_bar = [(1.0 - a_b) * (b / abs(a)) if a != 0 else b * delta for a, b, a_b in zip(A, B, A_bar)]

        h = [0.0] * state_dim
        outputs: List[float] = []

        for u in inputs:
            for i in range(state_dim):
                h[i] = A_bar[i] * h[i] + B_bar[i] * u
            y = sum(C[i] * h[i] for i in range(state_dim))
            outputs.append(y)

        return {
            "outputs": outputs,
            "final_state": h,
            "state_dim": state_dim,
        }
