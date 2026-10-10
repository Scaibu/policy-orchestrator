from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoPagedAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-118
      name: NnAlgoPagedAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - paged_attention
        - vllm
        - memory_management
      inputs:
        type: object
        required:
          - block_table
          - physical_blocks
          - query_vector
          - block_size
        properties:
          block_table:
            type: array
            items:
              type: integer
            description: Sequence logical-to-physical block mapping indices.
          physical_blocks:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Global physical memory block pool of shape (num_blocks, block_size, d_k).
          query_vector:
            type: array
            items:
              type: number
            description: Current query vector of shape (d_k).
          block_size:
            type: integer
            default: 16
            description: Number of tokens per block page.
      outputs:
        type: object
        required:
          - gathered_keys
          - total_tokens
          - fragmentation_loss
        properties:
          gathered_keys:
            type: array
            items:
              type: array
              items:
                type: number
            description: Non-contiguous physical keys gathered into sequence order.
          total_tokens:
            type: integer
            description: Total resolved token count.
          fragmentation_loss:
            type: number
            description: Memory fragmentation waste fraction.
      parameters: {}
      input_assumptions:
        - all block indices in block_table are valid
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact page indexing"
      uses_model: false
      complexity:
        variables:
          B: number of allocated blocks
          S: block_size
          d: head_dim
        time_worst: O(B * S * d)
        time_typical: O(B * S * d)
        space: O(B * S * d)
      preconditions:
        - len(input.block_table) > 0 and len(input.physical_blocks) > 0
      postconditions:
        - len(output.gathered_keys) == len(input.block_table) * input.block_size
      certificate: "Gathered keys correctly index physical blocks according to block_table"
      compatible_adapters:
        - ADAPTER-VLLM-PAGED-ATTN
      related_algos:
        - ALGO-NN-117
      references:
        - "https://arxiv.org/abs/2309.06180"
    ---
    """

    @staticmethod
    def forward(
        block_table: List[int],
        physical_blocks: List[List[List[float]]],
        query_vector: List[float],
        block_size: int = 16,
    ) -> Dict[str, Any]:
        if not block_table or not physical_blocks or not query_vector:
            raise ValueError("Precondition failed: inputs must be non-empty")

        gathered: List[List[float]] = []
        for p_idx in block_table:
            if not (0 <= p_idx < len(physical_blocks)):
                raise ValueError(f"Precondition failed: invalid physical block index {p_idx}")
            block = physical_blocks[p_idx]
            for tok_vec in block:
                gathered.append(tok_vec[:])

        total_toks = len(gathered)
        allocated_capacity = len(block_table) * block_size
        frag_loss = 1.0 - (float(total_toks) / float(allocated_capacity)) if allocated_capacity > 0 else 0.0

        return {
            "gathered_keys": gathered,
            "total_tokens": total_toks,
            "fragmentation_loss": max(0.0, frag_loss),
        }
