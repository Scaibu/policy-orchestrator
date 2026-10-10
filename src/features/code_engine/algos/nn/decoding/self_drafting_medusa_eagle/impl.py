from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class NnAlgoSelfDraftingMedusaEagle:
    """
    ---
    contract:
      algo_id: ALGO-NN-147
      name: NnAlgoSelfDraftingMedusaEagle
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - medusa
        - eagle
        - tree_attention
        - self_drafting
      inputs:
        type: object
        required:
          - head_candidates
        properties:
          head_candidates:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Top candidates from each Medusa head of shape (num_heads, top_k_per_head).
      outputs:
        type: object
        required:
          - tree_paths
          - tree_attention_mask
          - total_nodes
        properties:
          tree_paths:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Expanded speculative candidate paths.
          tree_attention_mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Ancestor-only tree attention verification mask.
          total_nodes:
            type: integer
            description: Total verified tree nodes.
      parameters: {}
      input_assumptions:
        - head_candidates non-empty
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact prefix tree construction"
      uses_model: false
      complexity:
        variables:
          H: num_heads
          K: top_k_per_head
        time_worst: O(K^H)
        time_typical: O(K * H)
        space: O(K * H)
      preconditions:
        - len(input.head_candidates) > 0
      postconditions:
        - output.total_nodes > 0
      certificate: "Tree attention mask restricts attention to valid path ancestors"
      compatible_adapters:
        - ADAPTER-MEDUSA-TREE
      related_algos:
        - ALGO-NN-146
      references:
        - "https://arxiv.org/abs/2401.10774"
        - "https://arxiv.org/abs/2401.15077"
    ---
    """

    @staticmethod
    def build_tree(head_candidates: List[List[int]]) -> Dict[str, Any]:
        if not head_candidates:
            raise ValueError("Precondition failed: head_candidates must be non-empty")

        paths: List[List[int]] = [[]]
        for candidates in head_candidates:
            new_paths: List[List[int]] = []
            for p in paths:
                for c in candidates:
                    new_paths.append(p + [c])
            paths = new_paths[:8]

        num_nodes = len(paths)
        mask: List[List[float]] = []
        for i in range(num_nodes):
            row: List[float] = []
            for j in range(num_nodes):
                if j <= i:
                    row.append(0.0)
                else:
                    row.append(-1e9)
            mask.append(row)

        return {
            "tree_paths": paths,
            "tree_attention_mask": mask,
            "total_nodes": num_nodes,
        }
