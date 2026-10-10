from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoEncoderDecoderTransformers:
    """
    ---
    contract:
      algo_id: ALGO-NN-113
      name: NnAlgoEncoderDecoderTransformers
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - encoder_decoder
        - cross_attention
        - sequence_to_sequence
      inputs:
        type: object
        required:
          - encoder_hidden_states
          - decoder_hidden_states
        properties:
          encoder_hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoder representation matrix of shape (src_len, d_model).
          decoder_hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Decoder query states of shape (tgt_len, d_model).
          cross_mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Optional cross-attention mask of shape (tgt_len, src_len).
      outputs:
        type: object
        required:
          - cross_attention_output
          - cross_weights
        properties:
          cross_attention_output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cross-attended representations of shape (tgt_len, d_model).
          cross_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cross-attention alignment weights.
      parameters: {}
      input_assumptions:
        - encoder and decoder feature dimensions match
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical bounds"
      uses_model: false
      complexity:
        variables:
          T_src: src_len
          T_tgt: tgt_len
          d: d_model
        time_worst: O(T_tgt * T_src * d)
        time_typical: O(T_tgt * T_src * d)
        space: O(T_tgt * T_src + T_tgt * d)
      preconditions:
        - len(input.encoder_hidden_states) > 0 and len(input.decoder_hidden_states) > 0
        - len(input.encoder_hidden_states[0]) == len(input.decoder_hidden_states[0])
      postconditions:
        - len(output.cross_attention_output) == len(input.decoder_hidden_states)
        - len(output.cross_attention_output[0]) == len(input.decoder_hidden_states[0])
      certificate: "Output aligns decoder tokens with encoder representations"
      compatible_adapters:
        - ADAPTER-SEQ2SEQ-CROSS-ATTN
      related_algos:
        - ALGO-NN-101
        - ALGO-NN-131
      references:
        - "https://arxiv.org/abs/1706.03762"
    ---
    """

    @staticmethod
    def forward(
        encoder_hidden_states: List[List[float]],
        decoder_hidden_states: List[List[float]],
        cross_mask: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if not encoder_hidden_states or not decoder_hidden_states:
            raise ValueError("Precondition failed: inputs must be non-empty")
        if len(encoder_hidden_states[0]) != len(decoder_hidden_states[0]):
            raise ValueError("Precondition failed: feature dimensions must match")

        res = NnAlgoScaledDotProductAttention.forward(
            query=decoder_hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            mask=cross_mask,
        )

        return {
            "cross_attention_output": res["output"],
            "cross_weights": res["attention_weights"],
        }
