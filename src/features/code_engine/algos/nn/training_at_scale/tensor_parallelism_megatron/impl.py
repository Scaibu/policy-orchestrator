from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoTensorParallelismMegatron:
    """
    ---
    contract:
      algo_id: ALGO-NN-235
      name: NnAlgoTensorParallelismMegatron
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - tensor_parallelism
        - megatron_lm
        - column_parallel
        - row_parallel
        - all_reduce
      inputs:
        type: object
        required:
          - input_vector
          - column_shards
          - row_shards
        properties:
          input_vector:
            type: array
            items:
              type: number
            description: Input activation vector x of dimension d_in.
          column_shards:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: List of N_tp column-parallel weight matrices of shape (d_mid / N_tp, d_in).
          row_shards:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: List of N_tp row-parallel weight matrices of shape (d_out, d_mid / N_tp).
      outputs:
        type: object
        required:
          - output_vector
          - shard_hidden_activations
          - tp_degree
          - communication_volume_elements
        properties:
          output_vector:
            type: array
            items:
              type: number
            description: Aggregated output activation vector y = sum_p W_row^p * GeLU(W_col^p * x).
          shard_hidden_activations:
            type: array
            items:
              type: array
              items:
                type: number
            description: Per-shard intermediate hidden representations computed on each TP rank.
          tp_degree:
            type: integer
            description: Tensor parallel degree N_tp (number of partition ranks).
          communication_volume_elements:
            type: integer
            description: Total elements communicated across all-reduce collective: 2 * (N_tp - 1) * d_out.
      complexity:
        time: O(d_in * d_mid + d_mid * d_out)
        space: O(d_mid + d_out)
      preconditions:
        - len(input.input_vector) > 0
        - len(input.column_shards) > 0
        - len(input.column_shards) == len(input.row_shards)
        - all(len(c) > 0 for c in input.column_shards)
        - all(len(r) > 0 for r in input.row_shards)
      postconditions:
        - len(output.output_vector) == len(input.row_shards[0])
        - output.tp_degree == len(input.column_shards)
        - output.communication_volume_elements >= 0
      certificate: "Column-parallel projection followed by row-parallel projection and all-reduce reproduces monolithic dense MLP"
      compatible_adapters:
        - ADAPTER-MEGATRON-COLUMN-ROW-PARALLEL
        - ADAPTER-TP-ALLREDUCE-ENGINE
      related_algos:
        - ALGO-NN-233
        - ALGO-NN-234
      references:
        - "https://arxiv.org/abs/1909.08053"
        - "https://arxiv.org/abs/2104.04473"
    ---
    """

    @classmethod
    def _gelu(cls, x: float) -> float:
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def forward(
        cls,
        input_vector: List[float],
        column_shards: List[List[List[float]]],
        row_shards: List[List[List[float]]],
    ) -> Dict[str, Any]:
        d_in = len(input_vector)
        tp_degree = len(column_shards)
        if d_in == 0 or tp_degree == 0:
            raise ValueError("Precondition failed: input_vector and column_shards cannot be empty")
        if len(row_shards) != tp_degree:
            raise ValueError("Precondition failed: column_shards and row_shards must match in TP degree")

        d_mid_shard = len(column_shards[0])
        d_out = len(row_shards[0])

        for p in range(tp_degree):
            if len(column_shards[p]) != d_mid_shard or any(len(row) != d_in for row in column_shards[p]):
                raise ValueError("Precondition failed: inconsistent column shard dimension")
            if len(row_shards[p]) != d_out or any(len(row) != d_mid_shard for row in row_shards[p]):
                raise ValueError("Precondition failed: inconsistent row shard dimension")

        # 1. Column-parallel projection on each TP rank: z^p = GeLU(W_col^p * x)
        shard_hidden: List[List[float]] = []
        partial_outputs: List[List[float]] = []

        for p in range(tp_degree):
            z_p: List[float] = []
            for i in range(d_mid_shard):
                dot_val = sum(column_shards[p][i][j] * input_vector[j] for j in range(d_in))
                z_p.append(cls._gelu(dot_val))
            shard_hidden.append(z_p)

            # 2. Row-parallel projection on each rank: y^p = W_row^p * z^p
            y_p: List[float] = []
            for i in range(d_out):
                dot_val = sum(row_shards[p][i][j] * z_p[j] for j in range(d_mid_shard))
                y_p.append(dot_val)
            partial_outputs.append(y_p)

        # 3. All-Reduce across TP ranks (Sum across ranks)
        output_vector = [0.0] * d_out
        for i in range(d_out):
            output_vector[i] = sum(partial_outputs[p][i] for p in range(tp_degree))

        # Ring All-Reduce communication volume in elements
        comm_elements = 2 * (tp_degree - 1) * d_out if tp_degree > 1 else 0

        return {
            "output_vector": output_vector,
            "shard_hidden_activations": shard_hidden,
            "tp_degree": tp_degree,
            "communication_volume_elements": comm_elements,
        }
