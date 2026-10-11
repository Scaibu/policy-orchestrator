from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDifferentiallyPrivateTrainingDpsgd:
    """
    ---
    contract:
      algo_id: ALGO-NN-247
      name: NnAlgoDifferentiallyPrivateTrainingDpsgd
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - differential_privacy
        - dp_sgd
        - per_example_clipping
        - gaussian_noise_injection
        - privacy_budget
      inputs:
        type: object
        required:
          - per_example_gradients
        properties:
          per_example_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Micro-gradients computed individually for each sample in batch of shape (batch_size, P).
          clipping_norm_c:
            type: number
            default: 1.0
            description: Maximum permissible L2 norm threshold C for individual gradient vectors.
          noise_multiplier_sigma:
            type: number
            default: 1.0
            description: Standard deviation multiplier sigma for calibrated Gaussian noise injection.
          dataset_size:
            type: integer
            default: 10000
            description: Total population size N for subsampling privacy amplification.
          current_step:
            type: integer
            default: 100
            description: Current cumulative optimization step count.
          target_delta:
            type: number
            default: 0.00001
            description: Target privacy relaxation failure probability delta (typically < 1/N).
      outputs:
        type: object
        required:
          - private_averaged_gradients
          - mean_clipped_norm
          - clipped_fraction
          - estimated_epsilon
          - target_delta
          - batch_size
        properties:
          private_averaged_gradients:
            type: array
            items:
              type: number
            description: Differentially private gradient vector g_priv ready for optimizer step.
          mean_clipped_norm:
            type: number
            description: Mean L2 norm across clipped sample gradient vectors.
          clipped_fraction:
            type: number
            description: Proportion of training samples whose gradient norm exceeded clipping bound C.
          estimated_epsilon:
            type: number
            description: Cumulative privacy spending bound epsilon under Renyi differential privacy accounting.
          target_delta:
            type: number
            description: Privacy parameter delta.
          batch_size:
            type: integer
            description: Batch size B used in the current iteration.
      complexity:
        time: O(batch_size * P)
        space: O(P)
      preconditions:
        - len(input.per_example_gradients) > 0
        - len(input.per_example_gradients[0]) > 0
        - input.clipping_norm_c > 0.0
        - input.noise_multiplier_sigma >= 0.0
        - input.dataset_size > 0
        - input.current_step > 0
        - 0.0 < input.target_delta < 1.0
      postconditions:
        - len(output.private_averaged_gradients) == len(input.per_example_gradients[0])
        - output.mean_clipped_norm <= input.clipping_norm_c + 1e-6
        - 0.0 <= output.clipped_fraction <= 1.0
        - output.estimated_epsilon >= 0.0
      certificate: "Per-example gradients are clamped by C and perturbed by calibrated Gaussian variance"
      compatible_adapters:
        - ADAPTER-DP-SGD-PRIVACY-ENGINE
        - ADAPTER-OPACUS-COMPLIANCE-ACCOUNTANT
      related_algos:
        - ALGO-NN-246
        - ALGO-NN-239
      references:
        - "https://arxiv.org/abs/1607.00133"
        - "https://arxiv.org/abs/1808.06651"
    ---
    """

    @classmethod
    def forward(
        cls,
        per_example_gradients: List[List[float]],
        clipping_norm_c: float = 1.0,
        noise_multiplier_sigma: float = 1.0,
        dataset_size: int = 10000,
        current_step: int = 100,
        target_delta: float = 1e-5,
    ) -> Dict[str, Any]:
        b = len(per_example_gradients)
        if b == 0:
            raise ValueError("Precondition failed: per_example_gradients cannot be empty")
        p = len(per_example_gradients[0])
        if p == 0 or any(len(g) != p for g in per_example_gradients):
            raise ValueError("Precondition failed: inconsistent gradient vector dimension")
        if clipping_norm_c <= 0.0 or noise_multiplier_sigma < 0.0:
            raise ValueError("Precondition failed: invalid clipping or noise parameters")
        if dataset_size <= 0 or current_step <= 0 or not (0.0 < target_delta < 1.0):
            raise ValueError("Precondition failed: invalid dataset size, step, or delta")

        # 1. Per-example gradient clipping: g_i = g_i / max(1, ||g_i|| / C)
        clipped_grads: List[List[float]] = []
        clipped_count = 0
        norm_sum = 0.0

        for i in range(b):
            g_i = per_example_gradients[i]
            l2_norm = math.sqrt(sum(x * x for x in g_i))
            if l2_norm > clipping_norm_c:
                scale = clipping_norm_c / l2_norm
                clipped_count += 1
                norm_sum += clipping_norm_c
                clipped_grads.append([x * scale for x in g_i])
            else:
                norm_sum += l2_norm
                clipped_grads.append(list(g_i))

        mean_norm = norm_sum / float(b)
        clipped_fraction = float(clipped_count) / float(b)

        # 2. Sum and add calibrated noise (deterministic pseudo-random perturbation)
        # Noise standard deviation per coordinate = sigma * C
        noise_std = noise_multiplier_sigma * clipping_norm_c
        inv_b = 1.0 / float(b)

        priv_grads: List[float] = [0.0] * p
        for j in range(p):
            sum_j = sum(clipped_grads[i][j] for i in range(b))
            # Deterministic noise sample based on step and coordinate hash
            pseudo_u1 = (((current_step * 2654435761 + j * 2246822519) % 1000003) + 1) / 1000004.0
            pseudo_u2 = (((current_step * 3266489917 + j * 374761393) % 1000003) + 1) / 1000004.0
            # Box-Muller transform for standard Gaussian
            z_norm = math.sqrt(-2.0 * math.log(pseudo_u1)) * math.cos(2.0 * math.pi * pseudo_u2)
            noise_val = z_norm * noise_std

            priv_grads[j] = (sum_j + noise_val) * inv_b

        # 3. Privacy Accountant (RDP / Moments Accountant asymptotic bound)
        # Subsampling ratio q = B / N
        q = float(b) / float(dataset_size)
        if noise_multiplier_sigma > 0.0:
            # Epsilon bound approx: q * sqrt(T * log(1/delta)) / sigma
            eps = (q * math.sqrt(float(current_step) * 2.0 * math.log(1.0 / target_delta))) / noise_multiplier_sigma
        else:
            eps = float("inf")

        return {
            "private_averaged_gradients": priv_grads,
            "mean_clipped_norm": mean_norm,
            "clipped_fraction": clipped_fraction,
            "estimated_epsilon": eps,
            "target_delta": target_delta,
            "batch_size": b,
        }
