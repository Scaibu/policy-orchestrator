from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoEnergyBasedModels:
    """
    ---
    contract:
      algo_id: ALGO-NN-160
      name: NnAlgoEnergyBasedModels
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - energy_based_models
        - restricted_boltzmann_machine
        - contrastive_divergence
        - free_energy
      inputs:
        type: object
        required:
          - visible_input
          - weights
          - visible_bias
          - hidden_bias
        properties:
          visible_input:
            type: array
            items:
              type: number
            description: Data visible vector v_0 of dimension D_v in [0, 1].
          weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Bipartite coupling weight matrix W of shape (D_h, D_v).
          visible_bias:
            type: array
            items:
              type: number
            description: Visible bias vector a of dimension D_v.
          hidden_bias:
            type: array
            items:
              type: number
            description: Hidden bias vector b of dimension D_h.
      outputs:
        type: object
        required:
          - free_energy_data
          - free_energy_model
          - contrastive_divergence_loss
          - reconstructed_visible
          - reconstruction_error
          - weight_gradient_cd1
        properties:
          free_energy_data:
            type: number
            description: Analytical free energy F(v_0) of empirical data state.
          free_energy_model:
            type: number
            description: Analytical free energy F(v_1) of 1-step reconstructed negative state.
          contrastive_divergence_loss:
            type: number
            description: Energy margin surrogate loss F(v_0) - F(v_1).
          reconstructed_visible:
            type: array
            items:
              type: number
            description: One-step Gibbs reconstruction probabilities v_1 of dimension D_v.
          reconstruction_error:
            type: number
            description: Mean squared reconstruction discrepancy ||v_0 - v_1||^2 / D_v.
          weight_gradient_cd1:
            type: array
            items:
              type: array
              items:
                type: number
            description: Approximate CD-1 parameter gradient matrix <h_0 v_0^T> - <h_1 v_1^T>.
      parameters: {}
      input_assumptions:
        - visible_input has positive length D_v with values in [0, 1]
        - weights has shape (D_h, D_v) matching hidden_bias and visible_bias
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized softplus via log1p(exp)"
      uses_model: false
      complexity:
        variables:
          D_v: visible dimension
          D_h: hidden dimension
        time_worst: O(D_v * D_h)
        time_typical: O(D_v * D_h)
        space: O(D_v * D_h)
      preconditions:
        - len(input.visible_input) > 0
        - len(input.visible_bias) == len(input.visible_input)
        - len(input.weights) > 0 and len(input.weights[0]) == len(input.visible_input)
        - len(input.hidden_bias) == len(input.weights)
      postconditions:
        - len(output.reconstructed_visible) == len(input.visible_input)
        - len(output.weight_gradient_cd1) == len(input.weights)
        - len(output.weight_gradient_cd1[0]) == len(input.visible_input)
        - output.reconstruction_error >= 0.0
      certificate: "Reconstruction error measures mean squared Euclidean distance"
      compatible_adapters:
        - ADAPTER-EBM-ENERGY-SCORER
        - ADAPTER-RBM-UNSUPERVISED-PRETRAINER
      related_algos:
        - ALGO-NN-151
        - ALGO-NN-162
      references:
        - "https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf"
    ---
    """

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _softplus(x: float) -> float:
        # ln(1 + exp(x))
        if x > 30.0:
            return x
        if x < -30.0:
            return 0.0
        return math.log1p(math.exp(x))

    @classmethod
    def calculate_free_energy(
        cls,
        v: List[float],
        w: List[List[float]],
        a: List[float],
        b: List[float],
    ) -> float:
        # F(v) = - a^T v - sum_j softplus(W_{j, :}^T v + b_j)
        linear = sum(a[i] * v[i] for i in range(len(v)))
        hidden_sum = 0.0
        for j in range(len(b)):
            act = b[j] + sum(w[j][i] * v[i] for i in range(len(v)))
            hidden_sum += cls._softplus(act)
        return -linear - hidden_sum

    @classmethod
    def forward(
        cls,
        visible_input: List[float],
        weights: List[List[float]],
        visible_bias: List[float],
        hidden_bias: List[float],
    ) -> Dict[str, Any]:
        D_v = len(visible_input)
        D_h = len(hidden_bias)
        if D_v == 0 or D_h == 0:
            raise ValueError("Precondition failed: dimensions cannot be empty")
        if len(visible_bias) != D_v:
            raise ValueError("Precondition failed: visible_bias dimension mismatch")
        if len(weights) != D_h or any(len(r) != D_v for r in weights):
            raise ValueError("Precondition failed: weights dimension mismatch")

        # 1. Positive phase: h_0 = sigmoid(W * v_0 + b)
        h_0: List[float] = []
        for j in range(D_h):
            act = hidden_bias[j] + sum(weights[j][i] * visible_input[i] for i in range(D_v))
            h_0.append(cls._sigmoid(act))

        # 2. Reconstruction (Gibbs downward): v_1 = sigmoid(W^T * h_0 + a)
        v_1: List[float] = []
        for i in range(D_v):
            act = visible_bias[i] + sum(weights[j][i] * h_0[j] for j in range(D_h))
            v_1.append(cls._sigmoid(act))

        # 3. Negative phase: h_1 = sigmoid(W * v_1 + b)
        h_1: List[float] = []
        for j in range(D_h):
            act = hidden_bias[j] + sum(weights[j][i] * v_1[i] for i in range(D_v))
            h_1.append(cls._sigmoid(act))

        # 4. CD-1 Gradient: grad_W = h_0 v_0^T - h_1 v_1^T
        weight_grad: List[List[float]] = []
        for j in range(D_h):
            row_grad: List[float] = []
            for i in range(D_v):
                g = (h_0[j] * visible_input[i]) - (h_1[j] * v_1[i])
                row_grad.append(g)
            weight_grad.append(row_grad)

        fe_data = cls.calculate_free_energy(visible_input, weights, visible_bias, hidden_bias)
        fe_model = cls.calculate_free_energy(v_1, weights, visible_bias, hidden_bias)
        cd_loss = fe_data - fe_model

        recon_mse = sum((visible_input[i] - v_1[i]) ** 2 for i in range(D_v)) / float(D_v)

        return {
            "free_energy_data": fe_data,
            "free_energy_model": fe_model,
            "contrastive_divergence_loss": cd_loss,
            "reconstructed_visible": v_1,
            "reconstruction_error": recon_mse,
            "weight_gradient_cd1": weight_grad,
        }
