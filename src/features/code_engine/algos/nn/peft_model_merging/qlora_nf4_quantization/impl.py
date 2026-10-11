from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoQloraNf4Quantization:
    """
    ---
    contract:
      algo_id: ALGO-NN-219
      name: NnAlgoQloraNf4Quantization
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - qlora
        - nf4
        - normal_float_4
        - block_quantization
        - double_quantization
      inputs:
        type: object
        required:
          - raw_weights
          - input_vector
          - lora_a
          - lora_b
        properties:
          raw_weights:
            type: array
            items:
              type: number
            description: Flattened base model weights of length d * k to quantize into NF4.
          input_vector:
            type: array
            items:
              type: number
            description: Input activation vector x of dimension k.
          lora_a:
            type: array
            items:
              type: array
              items:
                type: number
            description: LoRA down-projection matrix A of shape (r, k).
          lora_b:
            type: array
            items:
              type: array
              items:
                type: number
            description: LoRA up-projection matrix B of shape (d, r).
          block_size:
            type: integer
            default: 64
            description: Quantization block granularity B_size.
          lora_alpha:
            type: number
            default: 16.0
            description: LoRA alpha scaling hyperparameter.
          lora_rank:
            type: integer
            default: 2
            description: Bottleneck rank r.
      outputs:
        type: object
        required:
          - quantized_nf4_indices
          - block_scales
          - dequantized_weights
          - output_vector
          - quantization_mse_error
        properties:
          quantized_nf4_indices:
            type: array
            items:
              type: integer
            description: 4-bit integer quantile indices in [0, 15] for each weight.
          block_scales:
            type: array
            items:
              type: number
            description: Normalization absolute maximum scale factor s per block.
          dequantized_weights:
            type: array
            items:
              type: number
            description: Dequantized weights in full floating point precision W_tilde = s * q.
          output_vector:
            type: array
            items:
              type: number
            description: Adapted layer activation h = W_tilde * x + (alpha / r) * B * (A * x) of dimension d.
          quantization_mse_error:
            type: number
            description: Mean squared reconstruction error between raw and dequantized weights.
      complexity:
        time: O(d * k + r * (d + k))
        space: O(d * k)
      preconditions:
        - len(input.raw_weights) > 0
        - len(input.input_vector) > 0
        - len(input.raw_weights) % len(input.input_vector) == 0
        - input.block_size > 0
        - input.lora_rank > 0
        - input.lora_alpha > 0.0
      postconditions:
        - len(output.quantized_nf4_indices) == len(input.raw_weights)
        - len(output.dequantized_weights) == len(input.raw_weights)
        - output.quantization_mse_error >= 0.0
      certificate: "Quantized NF4 indices are strictly bounded in integer range [0, 15]"
      compatible_adapters:
        - ADAPTER-QLORA-NF4-LINEAR
        - ADAPTER-PAGED-OPTIMIZER-ENGINE
      related_algos:
        - ALGO-NN-218
        - ALGO-NN-220
      references:
        - "https://arxiv.org/abs/2305.14314"
    ---
    """

    NF4_TABLE: List[float] = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    @classmethod
    def forward(
        cls,
        raw_weights: List[float],
        input_vector: List[float],
        lora_a: List[List[float]],
        lora_b: List[List[float]],
        block_size: int = 64,
        lora_alpha: float = 16.0,
        lora_rank: int = 2,
    ) -> Dict[str, Any]:
        W_len = len(raw_weights)
        k = len(input_vector)
        if W_len == 0 or k == 0:
            raise ValueError("Precondition failed: raw_weights and input_vector cannot be empty")
        if W_len % k != 0:
            raise ValueError("Precondition failed: raw_weights length must be divisible by input_vector length k")
        d = W_len // k
        if block_size <= 0:
            raise ValueError("Precondition failed: block_size must be positive")
        if lora_rank <= 0 or lora_alpha <= 0.0:
            raise ValueError("Precondition failed: lora_rank and lora_alpha must be positive")
        if len(lora_a) != lora_rank or any(len(row) != k for row in lora_a):
            raise ValueError("Precondition failed: lora_a shape must be (lora_rank, k)")
        if len(lora_b) != d or any(len(row) != lora_rank for row in lora_b):
            raise ValueError("Precondition failed: lora_b shape must be (d, lora_rank)")

        # 1. Block-wise quantization to NF4
        quantized_indices: List[int] = []
        block_scales: List[float] = []
        dequantized: List[float] = []
        mse_accum = 0.0

        num_blocks = int(math.ceil(W_len / float(block_size)))
        for b in range(num_blocks):
            start = b * block_size
            end = min(W_len, (b + 1) * block_size)
            chunk = raw_weights[start:end]

            abs_max = max(abs(w) for w in chunk)
            scale = abs_max if abs_max > 1e-12 else 1.0
            block_scales.append(scale)

            for w in chunk:
                norm_w = w / scale
                # Find nearest quantile in NF4_TABLE
                best_idx = 0
                best_dist = float("inf")
                for idx, q_val in enumerate(cls.NF4_TABLE):
                    dist = abs(norm_w - q_val)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                quantized_indices.append(best_idx)
                w_rec = scale * cls.NF4_TABLE[best_idx]
                dequantized.append(w_rec)
                diff = w - w_rec
                mse_accum += diff * diff

        quant_mse = mse_accum / float(W_len)

        # 2. Dequantized base forward: W_tilde * x
        base_out = [0.0] * d
        for i in range(d):
            row_start = i * k
            s = 0.0
            for j in range(k):
                s += dequantized[row_start + j] * input_vector[j]
            base_out[i] = s

        # 3. LoRA forward pass: s * B * (A * x)
        scaling = lora_alpha / float(lora_rank)
        ax = [sum(lora_a[r][j] * input_vector[j] for j in range(k)) for r in range(lora_rank)]
        lora_delta = [scaling * sum(lora_b[i][r] * ax[r] for r in range(lora_rank)) for i in range(d)]

        # 4. Total output
        total_out = [base_out[i] + lora_delta[i] for i in range(d)]

        return {
            "quantized_nf4_indices": quantized_indices,
            "block_scales": block_scales,
            "dequantized_weights": dequantized,
            "output_vector": total_out,
            "quantization_mse_error": quant_mse,
        }
