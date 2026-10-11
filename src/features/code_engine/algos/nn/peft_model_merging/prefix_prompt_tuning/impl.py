from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPrefixPromptTuning:
    """
    ---
    contract:
      algo_id: ALGO-NN-222
      name: NnAlgoPrefixPromptTuning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - prefix_tuning
        - prompt_tuning
        - soft_prompts
        - attention_prefix
      inputs:
        type: object
        required:
          - query_vector
          - key_vectors
          - value_vectors
          - prefix_keys
          - prefix_values
        properties:
          query_vector:
            type: array
            items:
              type: number
            description: Target attention query vector q of dimension d.
          key_vectors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Context sequence key vectors K of shape (L, d).
          value_vectors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Context sequence value vectors V of shape (L, d).
          prefix_keys:
            type: array
            items:
              type: array
              items:
                type: number
            description: Learnable virtual prefix key vectors K_prefix of shape (P, d).
          prefix_values:
            type: array
            items:
              type: array
              items:
                type: number
            description: Learnable virtual prefix value vectors V_prefix of shape (P, d).
      outputs:
        type: object
        required:
          - attention_output
          - attention_weights
          - prefix_attention_mass
        properties:
          attention_output:
            type: array
            items:
              type: number
            description: Context attention output vector Attention(q, [K_prefix; K], [V_prefix; V]) of dimension d.
          attention_weights:
            type: array
            items:
              type: number
            description: Softmax attention distribution across concatenated prefix and context positions of length P + L.
          prefix_attention_mass:
            type: number
            description: Cumulative attention probability mass allocated strictly to the virtual prefix positions in [0, 1].
      complexity:
        time: O((P + L) * d)
        space: O(P + L + d)
      preconditions:
        - len(input.query_vector) > 0
        - len(input.key_vectors) > 0
        - len(input.value_vectors) == len(input.key_vectors)
        - len(input.prefix_keys) > 0
        - len(input.prefix_values) == len(input.prefix_keys)
        - all(len(k) == len(input.query_vector) for k in input.key_vectors)
        - all(len(v) == len(input.query_vector) for v in input.value_vectors)
        - all(len(pk) == len(input.query_vector) for pk in input.prefix_keys)
        - all(len(pv) == len(input.query_vector) for pv in input.prefix_values)
      postconditions:
        - len(output.attention_output) == len(input.query_vector)
        - len(output.attention_weights) == len(input.prefix_keys) + len(input.key_vectors)
        - 0.0 <= output.prefix_attention_mass <= 1.0
        - abs(sum(output.attention_weights) - 1.0) < 1e-5
      certificate: "Attention weights across prefix and context sum to exactly 1.0"
      compatible_adapters:
        - ADAPTER-PREFIX-ATTENTION-LAYER
        - ADAPTER-PROMPT-TUNING-EMBEDDING
      related_algos:
        - ALGO-NN-001
        - ALGO-NN-218
        - ALGO-NN-221
      references:
        - "https://arxiv.org/abs/2101.00190"
        - "https://arxiv.org/abs/2104.08691"
    ---
    """

    @classmethod
    def forward(
        cls,
        query_vector: List[float],
        key_vectors: List[List[float]],
        value_vectors: List[List[float]],
        prefix_keys: List[List[float]],
        prefix_values: List[List[float]],
    ) -> Dict[str, Any]:
        d = len(query_vector)
        L = len(key_vectors)
        P = len(prefix_keys)
        if d == 0 or L == 0 or P == 0:
            raise ValueError("Precondition failed: inputs cannot be empty")
        if len(value_vectors) != L or len(prefix_values) != P:
            raise ValueError("Precondition failed: values lengths must match keys lengths")
        if (
            any(len(k) != d for k in key_vectors)
            or any(len(v) != d for v in value_vectors)
            or any(len(pk) != d for pk in prefix_keys)
            or any(len(pv) != d for pv in prefix_values)
        ):
            raise ValueError("Precondition failed: all vectors must match query dimension d")

        scale = 1.0 / math.sqrt(float(d))

        # 1. Concatenate prefix and context keys / values: total length P + L
        cat_keys = prefix_keys + key_vectors
        cat_values = prefix_values + value_vectors
        total_len = P + L

        # 2. Compute attention raw logits: q . k_i * scale
        raw_scores: List[float] = []
        for i in range(total_len):
            dot_prod = sum(query_vector[j] * cat_keys[i][j] for j in range(d))
            raw_scores.append(dot_prod * scale)

        # 3. Softmax over P + L positions
        max_s = max(raw_scores)
        exp_scores = [math.exp(s - max_s) for s in raw_scores]
        sum_exp = sum(exp_scores)
        attn_weights = [e / sum_exp for e in exp_scores]

        # 4. Compute weighted sum of values: Attention output of length d
        attn_out = [0.0] * d
        for i in range(total_len):
            w = attn_weights[i]
            for j in range(d):
                attn_out[j] += w * cat_values[i][j]

        # Prefix mass: sum of weights for first P positions
        prefix_mass = sum(attn_weights[:P])

        return {
            "attention_output": attn_out,
            "attention_weights": attn_weights,
            "prefix_attention_mass": prefix_mass,
        }
