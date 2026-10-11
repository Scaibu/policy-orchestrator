from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoFp8Training:
    """
    ---
    contract:
      algo_id: ALGO-NN-232
      name: NnAlgoFp8Training
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - fp8_training
        - e4m3
        - e5m2
        - delayed_scaling
        - quantization
      inputs:
        type: object
        required:
          - tensor_values
        properties:
          tensor_values:
            type: array
            items:
              type: number
            description: Activations, weights, or gradients to be cast to FP8.
          fp8_format:
            type: string
            default: e4m3
            enum: [e4m3, e5m2]
            description: FP8 specification: E4M3 for forward weights/activations; E5M2 for backward gradients.
          current_scale:
            type: number
            default: 1.0
            description: Precomputed scaling factor used for current casting step.
          history_max_abs:
            type: array
            items:
              type: number
            description: History window of past maximum absolute tensor values.
          scaling_mode:
            type: string
            default: delayed
            enum: [delayed, current]
            description: Scaling update policy: delayed uses historical amax; current uses current tensor amax.
          margin:
            type: number
            default: 0.0
            description: Log2 exponent margin headroom to avoid overflow.
      outputs:
        type: object
        required:
          - quantized_integers
          - dequantized_values
          - updated_scale
          - updated_history_max
          - quantization_snr_db
          - tensor_amax
        properties:
          quantized_integers:
            type: array
            items:
              type: integer
            description: Quantized signed integer representations in valid FP8 dynamic range.
          dequantized_values:
            type: array
            items:
              type: number
            description: Values restored to full precision via dequantization x_deq = q * scale.
          updated_scale:
            type: number
            description: Scaling multiplier computed for the next training iteration.
          updated_history_max:
            type: array
            items:
              type: number
            description: Updated rolling history of maximum absolute tensor values.
          quantization_snr_db:
            type: number
            description: Signal-to-Noise Ratio (SNR) in decibels quantifying quantization fidelity.
          tensor_amax:
            type: number
            description: Maximum absolute value in input tensor.
      complexity:
        time: O(len(tensor_values))
        space: O(len(tensor_values))
      preconditions:
        - len(input.tensor_values) > 0
        - input.current_scale > 0.0
      postconditions:
        - len(output.quantized_integers) == len(input.tensor_values)
        - len(output.dequantized_values) == len(input.tensor_values)
        - output.updated_scale > 0.0
        - output.tensor_amax >= 0.0
      certificate: "Quantized integers are strictly clamped within format limit (-448 to 448 for E4M3, -57344 to 57344 for E5M2)"
      compatible_adapters:
        - ADAPTER-FP8-SCALER
        - ADAPTER-TRANSFORMER-ENGINE-QUANTIZER
      related_algos:
        - ALGO-NN-231
        - ALGO-NN-250
      references:
        - "https://arxiv.org/abs/2209.05433"
        - "https://arxiv.org/abs/2206.02924"
    ---
    """

    @classmethod
    def forward(
        cls,
        tensor_values: List[float],
        fp8_format: str = "e4m3",
        current_scale: float = 1.0,
        history_max_abs: List[float] = [],
        scaling_mode: str = "delayed",
        margin: float = 0.0,
    ) -> Dict[str, Any]:
        n = len(tensor_values)
        if n == 0:
            raise ValueError("Precondition failed: tensor_values cannot be empty")
        if current_scale <= 0.0:
            raise ValueError("Precondition failed: current_scale must be positive")

        # E4M3 max finite value = 448.0; E5M2 max finite value = 57344.0
        max_fp8 = 448.0 if fp8_format == "e4m3" else 57344.0

        current_amax = max(abs(x) for x in tensor_values)
        current_amax = max(1e-12, current_amax)

        # Scale factor used for quantization
        if scaling_mode == "current":
            active_scale = max_fp8 / (current_amax * (2.0 ** margin))
        else:
            active_scale = current_scale

        quantized_integers: List[int] = []
        dequantized_values: List[float] = []
        noise_sq_sum = 0.0
        signal_sq_sum = 0.0

        inv_active_scale = 1.0 / active_scale

        for x in tensor_values:
            # Scaled representation: round to integer and clamp
            scaled_val = x * active_scale
            # Clamp to [-max_fp8, max_fp8]
            clamped_val = max(-max_fp8, min(max_fp8, scaled_val))
            q_int = int(round(clamped_val))
            quantized_integers.append(q_int)

            # Dequantize
            x_recon = float(q_int) * inv_active_scale
            dequantized_values.append(x_recon)

            err = x - x_recon
            noise_sq_sum += err * err
            signal_sq_sum += x * x

        # Signal-to-noise ratio in decibels
        if noise_sq_sum < 1e-12:
            snr_db = 100.0
        else:
            snr_db = 10.0 * math.log10(max(1e-12, signal_sq_sum) / noise_sq_sum)

        # Update history
        updated_history = list(history_max_abs) + [current_amax]
        if len(updated_history) > 16:
            updated_history = updated_history[-16:]

        # Next scale calculation using delayed history max
        hist_amax = max(updated_history)
        next_scale = max_fp8 / (hist_amax * (2.0 ** margin))

        return {
            "quantized_integers": quantized_integers,
            "dequantized_values": dequantized_values,
            "updated_scale": next_scale,
            "updated_history_max": updated_history,
            "quantization_snr_db": snr_db,
            "tensor_amax": current_amax,
        }
