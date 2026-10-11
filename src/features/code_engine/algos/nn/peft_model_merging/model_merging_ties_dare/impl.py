from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoModelMergingTiesDare:
    """
    ---
    contract:
      algo_id: ALGO-NN-224
      name: NnAlgoModelMergingTiesDare
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - peft
        - model_merging
        - task_vectors
        - ties_merging
        - dare
        - slerp
      inputs:
        type: object
        required:
          - base_weights
          - task_weights
        properties:
          base_weights:
            type: array
            items:
              type: number
            description: Pretrained base model parameter weights of length d.
          task_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Specialized fine-tuned model parameter vectors of shape (M, d).
          merge_method:
            type: string
            default: ties
            enum: [ties, task_arithmetic, dare, slerp]
            description: Weight space merging algorithm.
          task_coefficients:
            type: array
            items:
              type: number
            description: Scaling multipliers lambda_k for each task vector of length M.
          top_k_fraction:
            type: number
            default: 0.5
            description: Trimming fraction for TIES (proportion of largest magnitude weights retained in (0, 1]).
          slerp_t:
            type: number
            default: 0.5
            description: Interpolation parameter t in [0, 1] for SLERP between two models.
      outputs:
        type: object
        required:
          - merged_weights
          - effective_task_vector
          - sign_consensus_ratio
        properties:
          merged_weights:
            type: array
            items:
              type: number
            description: Resulting unified model weights after weight-space synthesis of dimension d.
          effective_task_vector:
            type: array
            items:
              type: number
            description: Net displacement vector tau = W_merged - W_base.
          sign_consensus_ratio:
            type: number
            description: Proportion of parameter coordinates where task vectors shared unanimous directional sign.
      complexity:
        time: O(M * d log d)
        space: O(M * d)
      preconditions:
        - len(input.base_weights) > 0
        - len(input.task_weights) > 0
        - all(len(m) == len(input.base_weights) for m in input.task_weights)
        - 0.0 < input.top_k_fraction <= 1.0
        - 0.0 <= input.slerp_t <= 1.0
      postconditions:
        - len(output.merged_weights) == len(input.base_weights)
        - len(output.effective_task_vector) == len(input.base_weights)
        - 0.0 <= output.sign_consensus_ratio <= 1.0
      certificate: "If task_weights are identical to base_weights, merged_weights equals base_weights exactly"
      compatible_adapters:
        - ADAPTER-MODEL-MERGING-ENGINE
        - ADAPTER-TIES-CHECKPOINT-COMBINER
      related_algos:
        - ALGO-NN-218
        - ALGO-NN-220
      references:
        - "https://arxiv.org/abs/2306.01708"
        - "https://arxiv.org/abs/2311.03099"
        - "https://arxiv.org/abs/2212.04089"
    ---
    """

    @classmethod
    def forward(
        cls,
        base_weights: List[float],
        task_weights: List[List[float]],
        merge_method: str = "ties",
        task_coefficients: Optional[List[float]] = None,
        top_k_fraction: float = 0.5,
        slerp_t: float = 0.5,
    ) -> Dict[str, Any]:
        d = len(base_weights)
        M = len(task_weights)
        if d == 0 or M == 0:
            raise ValueError("Precondition failed: inputs cannot be empty")
        if any(len(row) != d for row in task_weights):
            raise ValueError("Precondition failed: each task weight must have length d")
        if not (0.0 < top_k_fraction <= 1.0):
            raise ValueError("Precondition failed: top_k_fraction must be in (0, 1]")
        if not (0.0 <= slerp_t <= 1.0):
            raise ValueError("Precondition failed: slerp_t must be in [0, 1]")

        coeffs = task_coefficients if task_coefficients is not None else [1.0 / float(M)] * M
        if len(coeffs) != M:
            raise ValueError("Precondition failed: task_coefficients length must match task_weights")

        # Compute raw task vectors: tau_m = task_weights[m] - base_weights
        task_vectors: List[List[float]] = []
        for m in range(M):
            tau_m = [task_weights[m][j] - base_weights[j] for j in range(d)]
            task_vectors.append(tau_m)

        merged: List[float] = [0.0] * d
        consensus_count = 0

        if merge_method == "slerp":
            if M < 2:
                # If only 1 model, return it
                merged = list(task_weights[0])
            else:
                # SLERP between first two models
                v1 = list(task_weights[0])
                v2 = list(task_weights[1])
                norm1 = math.sqrt(sum(x * x for x in v1))
                norm2 = math.sqrt(sum(x * x for x in v2))
                if norm1 < 1e-12 or norm2 < 1e-12:
                    # Linear fallback
                    merged = [(1.0 - slerp_t) * v1[j] + slerp_t * v2[j] for j in range(d)]
                else:
                    u1 = [x / norm1 for x in v1]
                    u2 = [x / norm2 for x in v2]
                    dot = sum(u1[j] * u2[j] for j in range(d))
                    dot = max(-1.0, min(1.0, dot))
                    omega = math.acos(dot)
                    if abs(omega) < 1e-6 or abs(math.sin(omega)) < 1e-6:
                        merged = [(1.0 - slerp_t) * v1[j] + slerp_t * v2[j] for j in range(d)]
                    else:
                        sin_omega = math.sin(omega)
                        scale1 = math.sin((1.0 - slerp_t) * omega) / sin_omega
                        scale2 = math.sin(slerp_t * omega) / sin_omega
                        target_norm = (1.0 - slerp_t) * norm1 + slerp_t * norm2
                        u_res = [scale1 * u1[j] + scale2 * u2[j] for j in range(d)]
                        merged = [u_res[j] * target_norm for j in range(d)]

        elif merge_method == "task_arithmetic":
            # Simple linear task arithmetic: W_base + sum_m lambda_m * tau_m
            for j in range(d):
                delta_j = sum(coeffs[m] * task_vectors[m][j] for m in range(M))
                merged[j] = base_weights[j] + delta_j

        elif merge_method == "ties":
            # 1. Trimming: keep top_k_fraction largest magnitude values per task vector
            k_keep = max(1, min(d, int(math.ceil(top_k_fraction * d))))
            trimmed_tasks: List[List[float]] = []
            for m in range(M):
                tau = task_vectors[m]
                mag_indices = sorted(range(d), key=lambda j: abs(tau[j]), reverse=True)
                keep_set = set(mag_indices[:k_keep])
                trimmed = [tau[j] if j in keep_set else 0.0 for j in range(d)]
                trimmed_tasks.append(trimmed)

            # 2. Elect sign & Disjoint merge
            for j in range(d):
                total_sign_mass = sum(trimmed_tasks[m][j] for m in range(M))
                majority_sign = 1.0 if total_sign_mass >= 0.0 else -1.0

                # Check sign consensus across non-zero elements
                signs = [1 if trimmed_tasks[m][j] > 0 else (-1 if trimmed_tasks[m][j] < 0 else 0) for m in range(M)]
                non_zero_signs = [s for s in signs if s != 0]
                if non_zero_signs and all(s == non_zero_signs[0] for s in non_zero_signs):
                    consensus_count += 1

                # Disjoint merge: average only agreeing values
                agreeing_vals: List[float] = []
                for m in range(M):
                    val = trimmed_tasks[m][j]
                    if (val > 0.0 and majority_sign > 0.0) or (val < 0.0 and majority_sign < 0.0):
                        agreeing_vals.append(val)

                if agreeing_vals:
                    delta_j = sum(agreeing_vals) / float(len(agreeing_vals))
                else:
                    delta_j = 0.0
                merged[j] = base_weights[j] + delta_j

        else:  # DARE
            drop_rate = 1.0 - top_k_fraction
            rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
            for j in range(d):
                delta_j = sum(coeffs[m] * (task_vectors[m][j] * rescale) for m in range(M))
                merged[j] = base_weights[j] + delta_j

        # Telemetry
        effective_task = [merged[j] - base_weights[j] for j in range(d)]
        consensus_ratio = consensus_count / float(d) if merge_method == "ties" else 1.0

        return {
            "merged_weights": merged,
            "effective_task_vector": effective_task,
            "sign_consensus_ratio": consensus_ratio,
        }
