from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMamlMetaLearning:
    """
    ---
    contract:
      algo_id: ALGO-NN-178
      name: NnAlgoMamlMetaLearning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - meta_learning
        - maml
        - few_shot_adaptation
        - bilevel_optimization
      inputs:
        type: object
        required:
          - initial_weights_theta
          - task_support_gradients
          - task_query_gradients
        properties:
          initial_weights_theta:
            type: array
            items:
              type: number
            description: Shared meta-initialization parameter vector theta of dimension P.
          task_support_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Inner-loop support task gradients of shape (B_tasks, P).
          task_query_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Outer-loop evaluated query task gradients of shape (B_tasks, P).
          alpha_inner_lr:
            type: number
            default: 0.01
            minimum: 0.00001
            description: Inner-loop task adaptation step size alpha.
          beta_outer_lr:
            type: number
            default: 0.001
            minimum: 0.00001
            description: Outer-loop meta-optimization step size beta.
      outputs:
        type: object
        required:
          - updated_meta_weights
          - adapted_task_weights
          - meta_gradient_norm
          - mean_inner_gradient_norm
        properties:
          updated_meta_weights:
            type: array
            items:
              type: number
            description: Meta-updated parameter vector theta - beta * G_meta of dimension P.
          adapted_task_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Task-adapted parameter vectors theta_tau_prime of shape (B_tasks, P).
          meta_gradient_norm:
            type: number
            description: Euclidean norm of the accumulated outer meta-gradient.
          mean_inner_gradient_norm:
            type: number
            description: Average Euclidean norm of inner support task gradients.
      parameters: {}
      input_assumptions:
        - initial_weights_theta has positive length P
        - task_support_gradients and task_query_gradients have matching shape (B_tasks, P) with B_tasks >= 1
        - alpha_inner_lr > 0 and beta_outer_lr > 0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact first-order meta-update"
      uses_model: false
      complexity:
        variables:
          T: number of tasks
          P: parameter dimension
        time_worst: O(T * P)
        time_typical: O(T * P)
        space: O(T * P)
      preconditions:
        - len(input.initial_weights_theta) > 0
        - len(input.task_support_gradients) > 0 and len(input.task_support_gradients) == len(input.task_query_gradients)
        - len(input.task_support_gradients[0]) == len(input.initial_weights_theta)
      postconditions:
        - len(output.updated_meta_weights) == len(input.initial_weights_theta)
        - len(output.adapted_task_weights) == len(input.task_support_gradients)
        - output.meta_gradient_norm >= 0.0
      certificate: "Outer meta-gradient strictly equals average of task query gradients"
      compatible_adapters:
        - ADAPTER-MAML-OPTIMIZER
        - ADAPTER-REPTILE-TRAINER
      related_algos:
        - ALGO-NN-177
      references:
        - "https://arxiv.org/abs/1703.03400"
    ---
    """

    @classmethod
    def forward(
        cls,
        initial_weights_theta: List[float],
        task_support_gradients: List[List[float]],
        task_query_gradients: List[List[float]],
        alpha_inner_lr: float = 0.01,
        beta_outer_lr: float = 0.001,
    ) -> Dict[str, Any]:
        P = len(initial_weights_theta)
        B_tasks = len(task_support_gradients)

        if P == 0 or B_tasks == 0:
            raise ValueError("Precondition failed: inputs cannot be empty")
        if len(task_query_gradients) != B_tasks:
            raise ValueError("Precondition failed: task count mismatch")
        if any(len(r) != P for r in task_support_gradients) or any(len(r) != P for r in task_query_gradients):
            raise ValueError("Precondition failed: gradient dimension mismatch with theta")
        if alpha_inner_lr <= 0.0 or beta_outer_lr <= 0.0:
            raise ValueError("Precondition failed: learning rates must be positive")

        # 1. Inner loop adaptation per task: theta_tau' = theta - alpha * g_supp_tau
        adapted_weights: List[List[float]] = []
        inner_norm_sum = 0.0

        for t_idx in range(B_tasks):
            t_supp_g = task_support_gradients[t_idx]
            task_w: List[float] = []
            g_sq = 0.0
            for p in range(P):
                val = initial_weights_theta[p] - alpha_inner_lr * t_supp_g[p]
                task_w.append(val)
                g_sq += t_supp_g[p] * t_supp_g[p]
            adapted_weights.append(task_w)
            inner_norm_sum += math.sqrt(g_sq)

        # 2. Outer loop accumulation: G_meta = (1 / B) * sum_tau g_query_tau
        meta_grad = [0.0] * P
        for t_idx in range(B_tasks):
            t_query_g = task_query_gradients[t_idx]
            for p in range(P):
                meta_grad[p] += t_query_g[p]

        inv_b = 1.0 / float(B_tasks)
        meta_sq_sum = 0.0
        updated_theta: List[float] = []

        for p in range(P):
            g_bar = meta_grad[p] * inv_b
            meta_sq_sum += g_bar * g_bar
            val = initial_weights_theta[p] - beta_outer_lr * g_bar
            updated_theta.append(val)

        return {
            "updated_meta_weights": updated_theta,
            "adapted_task_weights": adapted_weights,
            "meta_gradient_norm": math.sqrt(meta_sq_sum),
            "mean_inner_gradient_norm": inner_norm_sum * inv_b,
        }
