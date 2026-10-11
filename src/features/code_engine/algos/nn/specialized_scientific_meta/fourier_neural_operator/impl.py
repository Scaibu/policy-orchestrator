from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoFourierNeuralOperator:
    """
    ---
    contract:
      algo_id: ALGO-NN-183
      name: NnAlgoFourierNeuralOperator
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - scientific_ml
        - fno
        - fourier_neural_operator
        - operator_learning
        - spectral_convolution
      inputs:
        type: object
        required:
          - spatial_signal
          - weights_real
          - weights_imag
          - local_weight
          - num_modes
        properties:
          spatial_signal:
            type: array
            items:
              type: number
            description: 1D spatial discrete field signal of length N_spatial.
          weights_real:
            type: array
            items:
              type: number
            description: Real part of complex Fourier weights R_real of length num_modes.
          weights_imag:
            type: array
            items:
              type: number
            description: Imaginary part of complex Fourier weights R_imag of length num_modes.
          local_weight:
            type: number
            description: Linear skip weight W for local spatial shortcut W * v(x).
          num_modes:
            type: integer
            minimum: 1
            description: Maximum frequency modes to retain k_max <= N_spatial // 2.
      outputs:
        type: object
        required:
          - transformed_signal
          - spectral_energy_retained
          - total_input_power
          - high_frequency_truncated_power
        properties:
          transformed_signal:
            type: array
            items:
              type: number
            description: Output spatial field v_{t+1}(x) = ReLU(W * v(x) + IDFT(R * DFT(v))).
          spectral_energy_retained:
            type: number
            description: Spectral energy present in the lowest num_modes Fourier modes.
          total_input_power:
            type: number
            description: Total signal energy sum_n x[n]^2 across all spatial grid points.
          high_frequency_truncated_power:
            type: number
            description: Discarded high-frequency spectral energy beyond num_modes.
      parameters: {}
      input_assumptions:
        - spatial_signal has length N >= 2
        - weights_real and weights_imag have matching length num_modes
        - 1 <= num_modes <= N // 2
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact discrete Fourier transform within float tolerance"
      uses_model: false
      complexity:
        variables:
          N: spatial grid resolution
          K: number of Fourier modes
        time_worst: O(N * K)
        time_typical: O(N * K)
        space: O(N + K)
      preconditions:
        - len(input.spatial_signal) >= 2
        - input.num_modes >= 1 and input.num_modes <= len(input.spatial_signal) // 2
        - len(input.weights_real) == input.num_modes and len(input.weights_imag) == input.num_modes
      postconditions:
        - len(output.transformed_signal) == len(input.spatial_signal)
        - output.spectral_energy_retained >= 0.0
        - output.total_input_power >= 0.0
      certificate: "Spectral convolution operates strictly on lowest num_modes frequencies"
      compatible_adapters:
        - ADAPTER-FNO-SPECTRAL-LAYER
        - ADAPTER-PDE-SURROGATE-SOLVER
      related_algos:
        - ALGO-NN-182
      references:
        - "https://arxiv.org/abs/2010.08895"
    ---
    """

    @classmethod
    def forward(
        cls,
        spatial_signal: List[float],
        weights_real: List[float],
        weights_imag: List[float],
        local_weight: float,
        num_modes: int,
    ) -> Dict[str, Any]:
        N = len(spatial_signal)
        if N < 2:
            raise ValueError("Precondition failed: spatial_signal must have length >= 2")
        if not (1 <= num_modes <= N // 2):
            raise ValueError(f"Precondition failed: num_modes {num_modes} out of bounds [1, {N // 2}]")
        if len(weights_real) != num_modes or len(weights_imag) != num_modes:
            raise ValueError("Precondition failed: weights dimension mismatch with num_modes")

        # 1. Compute 1D Discrete Fourier Transform (DFT) for modes k in [0, num_modes - 1]
        # X[k] = sum_{n=0}^{N-1} x[n] * exp(-i * 2 * pi * k * n / N)
        retained_energy = 0.0
        two_pi_inv_n = 2.0 * math.pi / float(N)

        dft_real: List[float] = []
        dft_imag: List[float] = []

        for k in range(num_modes):
            re = 0.0
            im = 0.0
            for n in range(N):
                angle = two_pi_inv_n * float(k * n)
                re += spatial_signal[n] * math.cos(angle)
                im -= spatial_signal[n] * math.sin(angle)
            dft_real.append(re)
            dft_imag.append(im)
            retained_energy += (re * re + im * im) / float(N)

        # 2. Complex multiplication with learned spectral weights:
        # Y[k] = (W_re + i * W_im) * (X_re + i * X_im)
        # Y_re = W_re * X_re - W_im * X_im
        # Y_im = W_re * X_im + W_im * X_re
        y_real: List[float] = []
        y_imag: List[float] = []

        for k in range(num_modes):
            w_r = weights_real[k]
            w_i = weights_imag[k]
            x_r = dft_real[k]
            x_i = dft_imag[k]

            y_real.append(w_r * x_r - w_i * x_i)
            y_imag.append(w_r * x_i + w_i * x_r)

        # 3. Inverse DFT back to spatial domain (IDFT on truncated modes)
        # idft[n] = (1 / N) * [ Y[0] + 2 * sum_{k=1}^{K-1} (Y_re * cos + Y_im * sin) ]
        inv_n = 1.0 / float(N)
        idft_out: List[float] = []

        for n in range(N):
            val = y_real[0]  # k = 0 DC component
            for k in range(1, num_modes):
                angle = two_pi_inv_n * float(k * n)
                # Symmetrical positive and negative frequency contribution: 2 * Real(Y * exp(i*angle))
                val += 2.0 * (y_real[k] * math.cos(angle) - y_imag[k] * math.sin(angle))
            idft_out.append(val * inv_n)

        # 4. Combine with linear shortcut: v_{next} = ReLU(W * v + IDFT(Y))
        transformed: List[float] = []
        total_power = 0.0
        for n in range(N):
            x_n = spatial_signal[n]
            total_power += x_n * x_n
            lin_val = local_weight * x_n + idft_out[n]
            transformed.append(max(0.0, lin_val))  # ReLU activation

        trunc_power = max(0.0, total_power - retained_energy)

        return {
            "transformed_signal": transformed,
            "spectral_energy_retained": retained_energy,
            "total_input_power": total_power,
            "high_frequency_truncated_power": trunc_power,
        }
