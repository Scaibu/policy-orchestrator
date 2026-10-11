from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDataParallelismAllreduce:
    """
    ---
    contract:
      algo_id: ALGO-NN-233
      name: NnAlgoDataParallelismAllreduce
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - distributed_training
        - data_parallelism
        - all_reduce
        - ring_all_reduce
        - collective_communication
      inputs:
        type: object
        required:
          - worker_gradients
        properties:
          worker_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Locally computed gradients from N workers of shape (num_workers, gradient_dim).
          topology:
            type: string
            default: ring
            enum: [ring, tree, hierarchical]
            description: Collective communication reduction topology.
          bytes_per_parameter:
            type: integer
            default: 2
            description: Precision byte footprint per parameter (2 for BF16/FP16, 4 for FP32).
      outputs:
        type: object
        required:
          - averaged_gradients
          - bytes_transferred_per_worker
          - communication_steps
          - gradient_variance_across_workers
          - worker_count
          - parameter_count
        properties:
          averaged_gradients:
            type: array
            items:
              type: number
            description: Globally averaged gradient vector applied by all workers.
          bytes_transferred_per_worker:
            type: number
            description: Total communication volume sent by each worker: 2 * (N-1)/N * total_bytes.
          communication_steps:
            type: integer
            description: Number of point-to-point ring transfer rounds: 2 * (N-1).
          gradient_variance_across_workers:
            type: number
            description: Mean cross-worker parameter gradient variance.
          worker_count:
            type: integer
            description: Total number of data-parallel worker ranks N.
          parameter_count:
            type: integer
            description: Length of parameter gradient vector G.
      complexity:
        time: O(N * G)
        space: O(G)
      preconditions:
        - len(input.worker_gradients) > 0
        - all(len(g) == len(input.worker_gradients[0]) for g in input.worker_gradients)
        - len(input.worker_gradients[0]) > 0
        - input.bytes_per_parameter > 0
      postconditions:
        - len(output.averaged_gradients) == len(input.worker_gradients[0])
        - output.worker_count == len(input.worker_gradients)
        - output.bytes_transferred_per_worker >= 0.0
        - output.gradient_variance_across_workers >= 0.0
      certificate: "Averaged gradients match arithmetic mean across all worker gradients"
      compatible_adapters:
        - ADAPTER-RING-ALLREDUCE-COLLECTIVE
        - ADAPTER-DDP-GRADIENT-SYNCHRONIZER
      related_algos:
        - ALGO-NN-234
        - ALGO-NN-235
      references:
        - "https://arxiv.org/abs/2006.15704"
        - "https://arxiv.org/abs/1802.05799"
    ---
    """

    @classmethod
    def forward(
        cls,
        worker_gradients: List[List[float]],
        topology: str = "ring",
        bytes_per_parameter: int = 2,
    ) -> Dict[str, Any]:
        num_workers = len(worker_gradients)
        if num_workers == 0:
            raise ValueError("Precondition failed: worker_gradients cannot be empty")
        grad_dim = len(worker_gradients[0])
        if grad_dim == 0 or any(len(row) != grad_dim for row in worker_gradients):
            raise ValueError("Precondition failed: all workers must supply gradients of equal non-zero dimension")
        if bytes_per_parameter <= 0:
            raise ValueError("Precondition failed: bytes_per_parameter must be positive")

        # 1. Compute arithmetic mean
        averaged_grads: List[float] = []
        total_var = 0.0
        inv_n = 1.0 / float(num_workers)

        for j in range(grad_dim):
            col_vals = [worker_gradients[i][j] for i in range(num_workers)]
            mean_j = sum(col_vals) * inv_n
            averaged_grads.append(mean_j)

            var_j = sum((v - mean_j) ** 2 for v in col_vals) * inv_n
            total_var += var_j

        mean_variance = total_var / float(grad_dim)

        # 2. Ring communication metrics
        total_param_bytes = float(grad_dim * bytes_per_parameter)
        if num_workers > 1:
            if topology == "ring":
                # Ring All-Reduce: 2 * (N - 1) / N * size
                bytes_per_worker = 2.0 * float(num_workers - 1) / float(num_workers) * total_param_bytes
                steps = 2 * (num_workers - 1)
            elif topology == "tree":
                # Tree All-Reduce: 2 * log2(N) steps
                steps = 2 * int(math.ceil(math.log2(float(num_workers))))
                bytes_per_worker = 2.0 * total_param_bytes
            else:
                # Hierarchical
                bytes_per_worker = 1.5 * total_param_bytes
                steps = num_workers
        else:
            bytes_per_worker = 0.0
            steps = 0

        return {
            "averaged_gradients": averaged_grads,
            "bytes_transferred_per_worker": bytes_per_worker,
            "communication_steps": steps,
            "gradient_variance_across_workers": mean_variance,
            "worker_count": num_workers,
            "parameter_count": grad_dim,
        }
