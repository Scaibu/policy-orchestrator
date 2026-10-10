from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoAlibiAttentionLinearBiases:
    """
    ---
    contract:
      algo_id: ALGO-NN-109
      name: NnAlgoAlibiAttentionLinearBiases
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - alibi
        - linear_bias
        - length_extrapolation
      inputs:
        type: object
        required:
          - num_heads
          - seq_len_q
          - seq_len_k
        properties:
          num_heads:
            type: integer
            minimum: 1
            description: Number of attention heads h.
          seq_len_q:
            type: integer
            minimum: 1
            description: Query sequence length N_q.
          seq_len_k:
            type: integer
            minimum: 1
            description: Key sequence length N_k.
      outputs:
        type: object
        required:
          - bias_matrices
          - slopes
        properties:
          bias_matrices:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Tensor of shape (num_heads, seq_len_q, seq_len_k).
          slopes:
            type: array
            items:
              type: number
            description: Head slope values m_h.
      parameters: {}
      input_assumptions:
        - num_heads >= 1, seq_len_q >= 1, seq_len_k >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact arithmetic"
      uses_model: false
      complexity:
        variables:
          H: num_heads
          N_q: seq_len_q
          N_k: seq_len_k
        time_worst: O(H * N_q * N_k)
        time_typical: O(H * N_q * N_k)
        space: O(H * N_q * N_k)
      preconditions:
        - input.num_heads >= 1 and input.seq_len_q >= 1 and input.seq_len_k >= 1
      postconditions:
        - len(output.bias_matrices) == input.num_heads
        - len(output.bias_matrices[0]) == input.seq_len_q
        - len(output.bias_matrices[0][0]) == input.seq_len_k
      certificate: "Diagonal entries are 0.0, off-diagonal negative penalties"
      compatible_adapters:
        - ADAPTER-ALIBI-ATTENTION
      related_algos:
        - ALGO-NN-108
        - ALGO-NN-110
      references:
        - "https://arxiv.org/abs/2108.12409"
    ---
    """

    @staticmethod
    def forward(num_heads: int, seq_len_q: int, seq_len_k: int) -> Dict[str, Any]:
        if num_heads < 1 or seq_len_q < 1 or seq_len_k < 1:
            raise ValueError("Precondition failed: dimensions must be positive integers")

        def get_slopes(n: int) -> List[float]:
            def get_slopes_power_of_2(count: int) -> List[float]:
                start = math.pow(2.0, -math.pow(2.0, -(math.log2(count) - 3)))
                ratio = start
                return [start * math.pow(ratio, i) for i in range(count)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            closest_power = 2 ** math.floor(math.log2(n))
            base_slopes = get_slopes_power_of_2(closest_power)
            extra_slopes = get_slopes_power_of_2(2 * closest_power)[0::2][: n - closest_power]
            return base_slopes + extra_slopes

        slopes = get_slopes(num_heads)
        bias_matrices: List[List[List[float]]] = []

        for m in slopes:
            mat: List[List[float]] = []
            for i in range(seq_len_q):
                row: List[float] = []
                for j in range(seq_len_k):
                    penalty = -m * float(abs(i - j))
                    row.append(penalty)
                mat.append(row)
            bias_matrices.append(mat)

        return {
            "bias_matrices": bias_matrices,
            "slopes": slopes,
        }
