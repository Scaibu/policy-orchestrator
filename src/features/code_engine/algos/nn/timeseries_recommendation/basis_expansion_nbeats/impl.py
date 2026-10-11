from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoBasisExpansionNbeats:
    """
    ---
    contract:
      algo_id: ALGO-NN-189
      name: NnAlgoBasisExpansionNbeats
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - time_series
        - forecasting
        - nbeats
        - basis_expansion
        - doubly_residual
      inputs:
        type: object
        required:
          - lookback_window
          - weights_theta_backcast
          - weights_theta_forecast
          - forecast_horizon
        properties:
          lookback_window:
            type: array
            items:
              type: number
            description: Input time-series lookback history of length L.
          weights_theta_backcast:
            type: array
            items:
              type: array
              items:
                type: number
            description: Linear projection weights for backcast coefficients theta_b of shape (P_b, L).
          weights_theta_forecast:
            type: array
            items:
              type: array
              items:
                type: number
            description: Linear projection weights for forecast coefficients theta_f of shape (P_f, L).
          forecast_horizon:
            type: integer
            minimum: 1
            description: Number of future time steps H to predict.
          basis_type:
            type: string
            default: polynomial
            enum:
              - polynomial
              - generic
            description: Functional basis expansion family (polynomial trend or linear generic).
      outputs:
        type: object
        required:
          - backcast
          - forecast
          - residual_lookback
          - forecast_energy
          - backcast_reconstruction_loss
        properties:
          backcast:
            type: array
            items:
              type: number
            description: Reconstructed lookback signal x_hat of length L.
          forecast:
            type: array
            items:
              type: number
            description: Predicted future horizon signal y_hat of length H.
          residual_lookback:
            type: array
            items:
              type: number
            description: Unexplained residual signal x - x_hat passed to downstream doubly-residual blocks.
          forecast_energy:
            type: number
            description: L2 norm energy of the generated forecast horizon.
          backcast_reconstruction_loss:
            type: number
            description: Mean squared error between input lookback and reconstructed backcast.
      complexity:
        time: O(L * P_b + H * P_f)
        space: O(L + H + P_b + P_f)
      preconditions:
        - len(input.lookback_window) >= 2
        - input.forecast_horizon >= 1
        - len(input.weights_theta_backcast) > 0 and len(input.weights_theta_backcast[0]) == len(input.lookback_window)
        - len(input.weights_theta_forecast) > 0 and len(input.weights_theta_forecast[0]) == len(input.lookback_window)
      postconditions:
        - len(output.backcast) == len(input.lookback_window)
        - len(output.forecast) == input.forecast_horizon
        - len(output.residual_lookback) == len(input.lookback_window)
        - output.forecast_energy >= 0.0
        - output.backcast_reconstruction_loss >= 0.0
      certificate: "Residual lookback satisfies residual[i] == lookback[i] - backcast[i]"
      compatible_adapters:
        - ADAPTER-NBEATS-BLOCK
        - ADAPTER-NHITS-HIERARCHICAL-INTERPOLATOR
      related_algos:
        - ALGO-NN-183
      references:
        - "https://arxiv.org/abs/1905.10437"
    ---
    """

    @classmethod
    def forward(
        cls,
        lookback_window: List[float],
        weights_theta_backcast: List[List[float]],
        weights_theta_forecast: List[List[float]],
        forecast_horizon: int,
        basis_type: str = "polynomial",
    ) -> Dict[str, Any]:
        L = len(lookback_window)
        if L < 2:
            raise ValueError("Precondition failed: lookback_window length must be >= 2")
        if forecast_horizon < 1:
            raise ValueError("Precondition failed: forecast_horizon must be >= 1")

        P_b = len(weights_theta_backcast)
        P_f = len(weights_theta_forecast)
        if P_b == 0 or any(len(r) != L for r in weights_theta_backcast):
            raise ValueError("Precondition failed: weights_theta_backcast shape mismatch")
        if P_f == 0 or any(len(r) != L for r in weights_theta_forecast):
            raise ValueError("Precondition failed: weights_theta_forecast shape mismatch")

        # 1. Project lookback to expansion parameters theta_b and theta_f
        theta_b: List[float] = []
        for p in range(P_b):
            val = sum(weights_theta_backcast[p][i] * lookback_window[i] for i in range(L))
            theta_b.append(val)

        theta_f: List[float] = []
        for p in range(P_f):
            val = sum(weights_theta_forecast[p][i] * lookback_window[i] for i in range(L))
            theta_f.append(val)

        # 2. Synthesize backcast x_hat = sum_p theta_b[p] * V_b[p](t)
        # For polynomial: V[p](t) = (t / L)^p for t in [0, L-1]
        backcast: List[float] = []
        for t in range(L):
            val = 0.0
            norm_t = float(t) / float(L - 1)
            for p in range(P_b):
                if basis_type == "polynomial":
                    basis_val = norm_t ** p
                else:
                    basis_val = math.cos(math.pi * float(p) * norm_t)
                val += theta_b[p] * basis_val
            backcast.append(val)

        # 3. Synthesize forecast y_hat = sum_p theta_f[p] * V_f[p](tau)
        # For polynomial: V[p](tau) = (tau / H)^p for tau in [0, H-1]
        forecast: List[float] = []
        for tau in range(forecast_horizon):
            val = 0.0
            norm_tau = float(tau) / float(max(1, forecast_horizon - 1))
            for p in range(P_f):
                if basis_type == "polynomial":
                    basis_val = (1.0 + norm_tau) ** p  # continuation of trend
                else:
                    basis_val = math.cos(math.pi * float(p) * (1.0 + norm_tau))
                val += theta_f[p] * basis_val
            forecast.append(val)

        # 4. Compute residual lookback and error metrics
        residual_lookback: List[float] = []
        mse_accum = 0.0
        for i in range(L):
            diff = lookback_window[i] - backcast[i]
            residual_lookback.append(diff)
            mse_accum += diff * diff
        mse_loss = mse_accum / float(L)

        forecast_energy = math.sqrt(sum(y * y for y in forecast))

        return {
            "backcast": backcast,
            "forecast": forecast,
            "residual_lookback": residual_lookback,
            "forecast_energy": forecast_energy,
            "backcast_reconstruction_loss": mse_loss,
        }
