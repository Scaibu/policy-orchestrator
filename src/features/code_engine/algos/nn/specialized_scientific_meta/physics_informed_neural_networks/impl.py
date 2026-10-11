from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPhysicsInformedNeuralNetworks:
    """
    ---
    contract:
      algo_id: ALGO-NN-182
      name: NnAlgoPhysicsInformedNeuralNetworks
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - scientific_ml
        - pinns
        - physics_informed
        - differential_equations
        - pde_residual
      inputs:
        type: object
        required:
          - pde_residuals
          - ic_predictions
          - ic_targets
          - bc_predictions
          - bc_targets
        properties:
          pde_residuals:
            type: array
            items:
              type: number
            description: Evaluated PDE residual errors f(x_i, t_i) at N_collocation points.
          ic_predictions:
            type: array
            items:
              type: number
            description: Model solution u(x_i, 0) at N_ic initial condition points.
          ic_targets:
            type: array
            items:
              type: number
            description: Ground truth initial condition values u_0(x_i) of length N_ic.
          bc_predictions:
            type: array
            items:
              type: number
            description: Model solution u(x_bc, t_i) at N_bc boundary condition points.
          bc_targets:
            type: array
            items:
              type: number
            description: Ground truth boundary condition values of length N_bc.
          weight_pde:
            type: number
            default: 1.0
            description: Loss multiplier for PDE residual term.
          weight_ic:
            type: number
            default: 10.0
            description: Loss multiplier for initial condition term.
          weight_bc:
            type: number
            default: 10.0
            description: Loss multiplier for boundary condition term.
      outputs:
        type: object
        required:
          - total_pinn_loss
          - pde_residual_loss
          - ic_loss
          - bc_loss
          - max_pde_violation
        properties:
          total_pinn_loss:
            type: number
            description: Weighted composite PINN objective w_f * L_pde + w_ic * L_ic + w_bc * L_bc.
          pde_residual_loss:
            type: number
            description: Mean squared PDE residual over collocation points.
          ic_loss:
            type: number
            description: Mean squared error over initial condition points.
          bc_loss:
            type: number
            description: Mean squared error over boundary condition points.
          max_pde_violation:
            type: number
            description: Maximum absolute PDE violation max_i |f(x_i, t_i)|.
      parameters: {}
      input_assumptions:
        - pde_residuals has length N_f >= 1
        - ic_predictions and ic_targets have matching length N_ic >= 1
        - bc_predictions and bc_targets have matching length N_bc >= 1
        - weights >= 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact residual mean square accumulation"
      uses_model: false
      complexity:
        variables:
          N_f: collocation points
          N_ic: initial points
          N_bc: boundary points
        time_worst: O(N_f + N_ic + N_bc)
        time_typical: O(N_f + N_ic + N_bc)
        space: O(1)
      preconditions:
        - len(input.pde_residuals) > 0
        - len(input.ic_predictions) > 0 and len(input.ic_predictions) == len(input.ic_targets)
        - len(input.bc_predictions) > 0 and len(input.bc_predictions) == len(input.bc_targets)
        - input.weight_pde >= 0.0 and input.weight_ic >= 0.0 and input.weight_bc >= 0.0
      postconditions:
        - output.total_pinn_loss >= 0.0
        - output.pde_residual_loss >= 0.0
        - output.max_pde_violation >= 0.0
      certificate: "Total loss strictly matches w_pde * L_pde + w_ic * L_ic + w_bc * L_bc"
      compatible_adapters:
        - ADAPTER-PINN-PDE-SOLVER
        - ADAPTER-NEURAL-OPERATOR-VALIDATOR
      related_algos:
        - ALGO-NN-179
        - ALGO-NN-183
      references:
        - "https://doi.org/10.1016/j.jcp.2018.10.045"
    ---
    """

    @classmethod
    def forward(
        cls,
        pde_residuals: List[float],
        ic_predictions: List[float],
        ic_targets: List[float],
        bc_predictions: List[float],
        bc_targets: List[float],
        weight_pde: float = 1.0,
        weight_ic: float = 10.0,
        weight_bc: float = 10.0,
    ) -> Dict[str, Any]:
        N_f = len(pde_residuals)
        N_ic = len(ic_predictions)
        N_bc = len(bc_predictions)

        if N_f == 0 or N_ic == 0 or N_bc == 0:
            raise ValueError("Precondition failed: residual sets cannot be empty")
        if len(ic_targets) != N_ic or len(bc_targets) != N_bc:
            raise ValueError("Precondition failed: target lengths must match predictions")
        if weight_pde < 0.0 or weight_ic < 0.0 or weight_bc < 0.0:
            raise ValueError("Precondition failed: weights must be non-negative")

        # 1. PDE residual loss: (1 / N_f) * sum |f|^2
        pde_sq_sum = 0.0
        max_violation = 0.0
        for r in pde_residuals:
            abs_r = abs(r)
            if abs_r > max_violation:
                max_violation = abs_r
            pde_sq_sum += r * r
        loss_pde = pde_sq_sum / float(N_f)

        # 2. Initial condition MSE: (1 / N_ic) * sum (u - u0)^2
        ic_sq_sum = 0.0
        for p, t in zip(ic_predictions, ic_targets):
            diff = p - t
            ic_sq_sum += diff * diff
        loss_ic = ic_sq_sum / float(N_ic)

        # 3. Boundary condition MSE: (1 / N_bc) * sum (u - u_bc)^2
        bc_sq_sum = 0.0
        for p, t in zip(bc_predictions, bc_targets):
            diff = p - t
            bc_sq_sum += diff * diff
        loss_bc = bc_sq_sum / float(N_bc)

        # 4. Total weighted objective
        total_loss = weight_pde * loss_pde + weight_ic * loss_ic + weight_bc * loss_bc

        return {
            "total_pinn_loss": total_loss,
            "pde_residual_loss": loss_pde,
            "ic_loss": loss_ic,
            "bc_loss": loss_bc,
            "max_pde_violation": max_violation,
        }
