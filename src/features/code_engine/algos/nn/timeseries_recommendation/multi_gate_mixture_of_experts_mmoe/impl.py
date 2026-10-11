from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMultiGateMixtureOfExpertsMmoe:
    """
    ---
    contract:
      algo_id: ALGO-NN-193
      name: NnAlgoMultiGateMixtureOfExpertsMmoe
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - recommendation
        - multi_task
        - mmoe
        - mixture_of_experts
        - multi_gate
      inputs:
        type: object
        required:
          - input_x
          - expert_weights
          - expert_biases
          - task_gate_weights
          - task_tower_weights
          - task_tower_biases
        properties:
          input_x:
            type: array
            items:
              type: number
            description: Shared feature vector x of dimension D.
          expert_weights:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Linear weights for E experts of shape (E, D_expert, D).
          expert_biases:
            type: array
            items:
              type: array
              items:
                type: number
            description: Biases for E experts of shape (E, D_expert).
          task_gate_weights:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Gating weights for K tasks of shape (K_tasks, E, D).
          task_tower_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Task tower prediction weights of shape (K_tasks, D_expert).
          task_tower_biases:
            type: array
            items:
              type: number
            description: Task tower prediction scalar biases of length K_tasks.
      outputs:
        type: object
        required:
          - task_predictions
          - task_gate_distributions
          - expert_representations
        properties:
          task_predictions:
            type: array
            items:
              type: number
            description: Calibrated prediction probabilities per task of length K_tasks in [0, 1].
          task_gate_distributions:
            type: array
            items:
              type: array
              items:
                type: number
            description: Softmax mixture coefficients over experts per task of shape (K_tasks, E).
          expert_representations:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hidden representations produced by each expert of shape (E, D_expert).
      complexity:
        time: O(E * D_expert * D + K * (E * D + E * D_expert))
        space: O(E * D_expert + K * E)
      preconditions:
        - len(input.input_x) > 0
        - len(input.expert_weights) > 0 and len(input.expert_biases) == len(input.expert_weights)
        - len(input.task_gate_weights) > 0 and len(input.task_tower_weights) == len(input.task_gate_weights)
        - len(input.task_tower_biases) == len(input.task_gate_weights)
      postconditions:
        - len(output.task_predictions) == len(input.task_gate_weights)
        - len(output.task_gate_distributions) == len(input.task_gate_weights)
        - len(output.expert_representations) == len(input.expert_weights)
        - all(0.0 <= p <= 1.0 for p in output.task_predictions)
      certificate: "Each task gate distribution is a valid probability simplex vector summing to 1.0"
      compatible_adapters:
        - ADAPTER-MMOE-MULTI-TASK-ROUTER
        - ADAPTER-JOINT-CTR-CVR-RANKER
      related_algos:
        - ALGO-NN-191
      references:
        - "https://doi.org/10.1145/3219819.3220007"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_x: List[float],
        expert_weights: List[List[List[float]]],
        expert_biases: List[List[float]],
        task_gate_weights: List[List[List[float]]],
        task_tower_weights: List[List[float]],
        task_tower_biases: List[float],
    ) -> Dict[str, Any]:
        D = len(input_x)
        if D == 0:
            raise ValueError("Precondition failed: input_x cannot be empty")

        E = len(expert_weights)
        if E == 0 or len(expert_biases) != E:
            raise ValueError("Precondition failed: expert weights/biases count mismatch")
        D_expert = len(expert_biases[0])
        if any(len(w) != D_expert or any(len(row) != D for row in w) for w in expert_weights):
            raise ValueError("Precondition failed: expert weights shape mismatch")

        K = len(task_gate_weights)
        if K == 0 or len(task_tower_weights) != K or len(task_tower_biases) != K:
            raise ValueError("Precondition failed: task count mismatch across gates and towers")
        if any(len(gw) != E or any(len(grow) != D for grow in gw) for gw in task_gate_weights):
            raise ValueError("Precondition failed: task_gate_weights shape mismatch")
        if any(len(tw) != D_expert for tw in task_tower_weights):
            raise ValueError("Precondition failed: task_tower_weights shape mismatch")

        # 1. Compute expert forward passes: expert_i = ReLU(W_i * x + b_i)
        expert_outputs: List[List[float]] = []
        for i in range(E):
            e_rep: List[float] = []
            for d_e in range(D_expert):
                val = sum(expert_weights[i][d_e][j] * input_x[j] for j in range(D)) + expert_biases[i][d_e]
                e_rep.append(max(0.0, val))  # ReLU activation
            expert_outputs.append(e_rep)

        # 2. For each task k, compute task gating and task prediction
        task_preds: List[float] = []
        task_gates: List[List[float]] = []

        for k in range(K):
            # Compute gate logits: g_logits = W_g^k * x
            g_logits: List[float] = []
            for i in range(E):
                logit = sum(task_gate_weights[k][i][j] * input_x[j] for j in range(D))
                g_logits.append(logit)

            # Softmax gate distribution over experts
            max_gl = max(g_logits)
            exps = [math.exp(gl - max_gl) for gl in g_logits]
            sum_exp = sum(exps)
            g_probs = [e / sum_exp for e in exps]
            task_gates.append(g_probs)

            # Task representation: u_k = sum_i g_k[i] * expert_i
            u_k = [0.0] * D_expert
            for i in range(E):
                p = g_probs[i]
                for d_e in range(D_expert):
                    u_k[d_e] += p * expert_outputs[i][d_e]

            # Tower prediction: y_k = sigmoid(W_tower^k * u_k + b_tower^k)
            t_logit = sum(task_tower_weights[k][d_e] * u_k[d_e] for d_e in range(D_expert)) + task_tower_biases[k]
            if t_logit >= 0:
                p_k = 1.0 / (1.0 + math.exp(-t_logit))
            else:
                ez = math.exp(t_logit)
                p_k = ez / (1.0 + ez)
            task_preds.append(p_k)

        return {
            "task_predictions": task_preds,
            "task_gate_distributions": task_gates,
            "expert_representations": expert_outputs,
        }
