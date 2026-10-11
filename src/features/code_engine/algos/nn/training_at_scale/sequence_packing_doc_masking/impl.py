from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class NnAlgoSequencePackingDocMasking:
    """
    ---
    contract:
      algo_id: ALGO-NN-230
      name: NnAlgoSequencePackingDocMasking
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - sequence_packing
        - document_masking
        - first_fit_decreasing
        - zero_padding
      inputs:
        type: object
        required:
          - document_lengths
          - max_sequence_length
        properties:
          document_lengths:
            type: array
            items:
              type: integer
            description: List of token lengths of incoming documents.
          max_sequence_length:
            type: integer
            description: Context window token capacity per packed sequence bin.
      outputs:
        type: object
        required:
          - packed_bins
          - bin_lengths
          - position_ids
          - document_boundaries
          - padding_efficiency
          - total_packed_sequences
        properties:
          packed_bins:
            type: array
            items:
              type: array
              items:
                type: integer
            description: List of document IDs packed into each sequence bin.
          bin_lengths:
            type: array
            items:
              type: integer
            description: Total token content length in each packed bin.
          position_ids:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Positional indices reset to 0 at the start of each packed document.
          document_boundaries:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Cumulative token end offsets for block-diagonal attention masking.
          padding_efficiency:
            type: number
            description: Ratio of valid document tokens to total allocated capacity across all bins.
          total_packed_sequences:
            type: integer
            description: Total number of allocated sequence bins.
      complexity:
        time: O(N log N + N * B)
        space: O(N + B * max_len)
      preconditions:
        - len(input.document_lengths) > 0
        - input.max_sequence_length > 0
        - all(0 < l <= input.max_sequence_length for l in input.document_lengths)
      postconditions:
        - all(l <= input.max_sequence_length for l in output.bin_lengths)
        - 0.0 < output.padding_efficiency <= 1.0
        - output.total_packed_sequences == len(output.packed_bins)
      certificate: "Packed document token lengths in every bin do not exceed max_sequence_length"
      compatible_adapters:
        - ADAPTER-DOCUMENT-PACKER
        - ADAPTER-BLOCK-DIAGONAL-MASK
      related_algos:
        - ALGO-NN-229
        - ALGO-NN-237
      references:
        - "https://arxiv.org/abs/2107.03374"
        - "https://arxiv.org/abs/2309.06180"
    ---
    """

    @classmethod
    def forward(
        cls,
        document_lengths: List[int],
        max_sequence_length: int,
    ) -> Dict[str, Any]:
        n_docs = len(document_lengths)
        if n_docs == 0:
            raise ValueError("Precondition failed: document_lengths cannot be empty")
        if max_sequence_length <= 0:
            raise ValueError("Precondition failed: max_sequence_length must be positive")
        if any(l <= 0 or l > max_sequence_length for l in document_lengths):
            raise ValueError("Precondition failed: document length must be in [1, max_sequence_length]")

        # First-Fit Decreasing (FFD) Bin Packing heuristic
        # Sort documents by descending length while preserving original doc index
        sorted_indices = sorted(range(n_docs), key=lambda i: document_lengths[i], reverse=True)

        bins_doc_ids: List[List[int]] = []
        bins_remaining: List[int] = []

        for doc_idx in sorted_indices:
            doc_len = document_lengths[doc_idx]
            placed = False
            for b_idx in range(len(bins_remaining)):
                if bins_remaining[b_idx] >= doc_len:
                    bins_doc_ids[b_idx].append(doc_idx)
                    bins_remaining[b_idx] -= doc_len
                    placed = True
                    break
            if not placed:
                # Open a new bin
                bins_doc_ids.append([doc_idx])
                bins_remaining.append(max_sequence_length - doc_len)

        # Generate position IDs and cumulative boundaries for each bin
        bin_lengths: List[int] = []
        position_ids: List[List[int]] = []
        document_boundaries: List[List[int]] = []
        total_valid_tokens = 0

        for b_idx, docs in enumerate(bins_doc_ids):
            b_positions: List[int] = []
            b_bounds: List[int] = []
            cur_offset = 0

            for d_id in docs:
                d_len = document_lengths[d_id]
                total_valid_tokens += d_len
                # Reset position IDs from 0 to d_len - 1
                b_positions.extend(range(d_len))
                cur_offset += d_len
                b_bounds.append(cur_offset)

            bin_lengths.append(cur_offset)
            position_ids.append(b_positions)
            document_boundaries.append(b_bounds)

        total_capacity = len(bins_doc_ids) * max_sequence_length
        efficiency = float(total_valid_tokens) / float(total_capacity)

        return {
            "packed_bins": bins_doc_ids,
            "bin_lengths": bin_lengths,
            "position_ids": position_ids,
            "document_boundaries": document_boundaries,
            "padding_efficiency": efficiency,
            "total_packed_sequences": len(bins_doc_ids),
        }
