from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDifferentiableNeuralMemory:
    """
    ---
    contract:
      algo_id: ALGO-NN-185
      name: NnAlgoDifferentiableNeuralMemory
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - differentiable_memory
        - ntm
        - memory_networks
        - soft_addressing
      inputs:
        type: object
        required:
          - memory_matrix
          - read_key
        properties:
          memory_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Memory slot matrix M of shape (N_slots, D_mem).
          read_key:
            type: array
            items:
              type: number
            description: Content-based query key k of dimension D_mem.
          key_strength_beta:
            type: number
            default: 1.0
            minimum: 0.0001
            description: Softmax sharpness temperature multiplier beta > 0.
          write_weights:
            type: array
            items:
              type: number
            description: Optional write addressing weights w_w of length N_slots.
          erase_vector:
            type: array
            items:
              type: number
            description: Optional erase vector e in [0, 1]^D_mem.
          add_vector:
            type: array
            items:
              type: number
            description: Optional add vector a of dimension D_mem.
      outputs:
        type: object
        required:
          - read_vector
          - read_weights
          - updated_memory_matrix
          - memory_frobenius_norm
        properties:
          read_vector:
            type: array
            items:
              type: number
            description: Retrieved memory content r = sum_i w_r[i] * M[i] of dimension D_mem.
          read_weights:
            type: array
            items:
              type: number
            description: Content-addressing softmax weights w_r of length N_slots.
          updated_memory_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Updated memory state after erase and add operations of shape (N_slots, D_mem).
          memory_frobenius_norm:
            type: number
            description: Frobenius norm of the memory matrix after read/write operations.
      parameters: {}
      input_assumptions:
        - memory_matrix has shape (N_slots, D_mem) with N_slots >= 1, D_mem >= 1
        - read_key has length D_mem
        - key_strength_beta > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized cosine similarity with eps 1e-8"
      uses_model: false
      complexity:
        variables:
          N: number of memory slots
          D: memory slot dimension
        time_worst: O(N * D)
        time_typical: O(N * D)
        space: O(N * D)
      preconditions:
        - len(input.memory_matrix) > 0 and len(input.memory_matrix[0]) > 0
        - len(input.read_key) == len(input.memory_matrix[0])
        - input.key_strength_beta > 0.0
      postconditions:
        - len(output.read_vector) == len(input.read_key)
        - len(output.read_weights) == len(input.memory_matrix)
        - len(output.updated_memory_matrix) == len(input.memory_matrix)
        - output.memory_frobenius_norm >= 0.0
      certificate: "Read vector satisfies convex combination: r = sum_i w_r[i] * M[i]"
      compatible_adapters:
        - ADAPTER-NTM-MEMORY-CONTROLLER
        - ADAPTER-DIFFERENTIABLE-KV-STORE
      related_algos:
        - ALGO-NN-184
      references:
        - "https://arxiv.org/abs/1410.5401"
    ---
    """

    @classmethod
    def forward(
        cls,
        memory_matrix: List[List[float]],
        read_key: List[float],
        key_strength_beta: float = 1.0,
        write_weights: List[float] | None = None,
        erase_vector: List[float] | None = None,
        add_vector: List[float] | None = None,
    ) -> Dict[str, Any]:
        N = len(memory_matrix)
        if N == 0 or len(memory_matrix[0]) == 0:
            raise ValueError("Precondition failed: memory_matrix cannot be empty")
        D = len(memory_matrix[0])

        if len(read_key) != D:
            raise ValueError("Precondition failed: read_key dimension mismatch")
        if key_strength_beta <= 0.0:
            raise ValueError("Precondition failed: key_strength_beta must be positive")

        eps = 1e-8

        # 1. Content-based cosine addressing: K(k, M[i])
        norm_k_sq = sum(x * x for x in read_key)
        norm_k = math.sqrt(norm_k_sq) + eps

        cos_scores: List[float] = []
        for i in range(N):
            row = memory_matrix[i]
            dot = sum(read_key[d] * row[d] for d in range(D))
            norm_m = math.sqrt(sum(row[d] * row[d] for d in range(D))) + eps
            cos_scores.append(key_strength_beta * (dot / (norm_k * norm_m)))

        # Softmax over slots
        max_score = max(cos_scores)
        exps = [math.exp(s - max_score) for s in cos_scores]
        sum_exp = sum(exps)
        w_r = [e / sum_exp for e in exps]

        # 2. Read vector: r = sum_i w_r[i] * M[i]
        read_vec = [0.0] * D
        for i in range(N):
            w = w_r[i]
            for d in range(D):
                read_vec[d] += w * memory_matrix[i][d]

        # 3. Optional Write: Erase + Add
        # M_new[i, d] = M[i, d] * (1 - w_w[i] * e[d]) + w_w[i] * a[d]
        updated_mem: List[List[float]] = []
        norm_sq = 0.0

        has_write = (
            write_weights is not None
            and erase_vector is not None
            and add_vector is not None
            and len(write_weights) == N
            and len(erase_vector) == D
            and len(add_vector) == D
        )

        for i in range(N):
            row_out: List[float] = []
            w_w = write_weights[i] if has_write else 0.0
            for d in range(D):
                orig_val = memory_matrix[i][d]
                if has_write:
                    e_d = max(0.0, min(1.0, erase_vector[d]))
                    a_d = add_vector[d]
                    val = orig_val * (1.0 - w_w * e_d) + w_w * a_d
                else:
                    val = orig_val
                row_out.append(val)
                norm_sq += val * val
            updated_mem.append(row_out)

        return {
            "read_vector": read_vec,
            "read_weights": w_r,
            "updated_memory_matrix": updated_mem,
            "memory_frobenius_norm": math.sqrt(norm_sq),
        }
