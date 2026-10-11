from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoCriticalBatchSizeNoiseScale:
    """
    ---
    contract:
      algo_id: ALGO-NN-244
      name: NnAlgoCriticalBatchSizeNoiseScale
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - critical_batch_size
        - gradient_noise_scale
        - batch_size_schedule
        - sample_efficiency
      inputs:
        type: object
        required:
          - per_sample_gradients
          - current_step
          - total_steps
        properties:
          per_sample_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Sample-level gradients of shape (B_sample, P).
          current_step:
            type: integer
            description: Current training step.
          total_steps:
            type: integer
            description: Total scheduled steps.
          min_batch_size:
            type: integer
            default: 32
            description: Minimum batch size floor during warmup.
          max_batch_size:
            type: integer
            default: 2048
            description: Maximum batch size ceiling at asymptote.
      outputs:
        type: object
        required:
          - estimated_noise_scale_b_noise
          - gradient_trace_covariance
          - mean_gradient_squared_norm
          - recommended_batch_size
          - compute_efficiency
        properties:
          estimated_noise_scale_b_noise:
            type: number
            description: Critical batch size B_noise = tr(Sigma) / ||G||^2.
          gradient_trace_covariance:
            type: number
            description: Total variance (trace of covariance) across sample gradients.
          mean_gradient_squared_norm:
            type: number
            description: Squared Euclidean norm of the expected gradient vector ||G||^2.
          recommended_batch_size:
            type: integer
            description: Recommended batch size dynamically clamped to [min_batch, max_batch].
          compute_efficiency:
            type: number
            description: Relative parallel compute efficiency E = 1 / (1 + B / B_noise).
      complexity:
        time: O(B_sample * P)
        space: O(P)
      preconditions:
        - len(input.per_sample_gradients) > 1
        - len(input.per_sample_gradients[0]) > 0
        - input.current_step >= 0
        - input.total_steps > 0
        - input.min_batch_size > 0
        - input.max_batch_size >= input.min_batch_size
      postconditions:
        - output.estimated_noise_scale_b_noise > 0.0
        - input.min_batch_size <= output.recommended_batch_size <= input.max_batch_size
        - 0.0 < output.compute_efficiency <= 1.0
      certificate: "B_noise equals trace of empirical sample covariance divided by squared norm of mean gradient"
      compatible_adapters:
        - ADAPTER-NOISE-SCALE-PROFILER
        - ADAPTER-DYNAMIC-BATCH-RAMP
      related_algos:
        - ALGO-NN-239
        - ALGO-NN-233
      references:
        - "https://arxiv.org/abs/1812.06162"
        - "https://arxiv.org/abs/2006.15704"
    ---
    """

    @classmethod
    def forward(
        cls,
        per_sample_gradients: List[List[float]],
        current_step: int,
        total_steps: int,
        min_batch_size: int = 32,
        max_batch_size: int = 2048,
    ) -> Dict[str, Any]:
        b_sample = len(per_sample_gradients)
        if b_sample <= 1:
            raise ValueError("Precondition failed: need at least 2 samples to compute covariance")
        p = len(per_sample_gradients[0])
        if p == 0 or any(len(g) != p for g in per_sample_gradients):
            raise ValueError("Precondition failed: inconsistent gradient dimension")
        if min_batch_size <= 0 or max_batch_size < min_batch_size:
            raise ValueError("Precondition failed: invalid batch bounds")

        inv_b = 1.0 / float(b_sample)

        # 1. Mean gradient vector G
        mean_g = [0.0] * p
        for i in range(b_sample):
            for j in range(p):
                mean_g[j] += per_sample_gradients[i][j] * inv_b

        # 2. Squared norm of mean gradient: ||G||^2
        mean_g_sq_norm = sum(g * g for g in mean_g)

        # 3. Trace of covariance: tr(Sigma) = sum_j Var(g_j)
        tr_cov = 0.0
        for j in range(p):
            var_j = sum((per_sample_gradients[i][j] - mean_g[j]) ** 2 for i in range(b_sample)) / float(b_sample - 1)
            tr_cov += var_j

        # Critical batch size B_noise = tr(Sigma) / max(eps, ||G||^2)
        b_noise = tr_cov / max(1e-12, mean_g_sq_norm)

        # Recommended batch size following warm-up ramp schedule toward B_noise
        progress = min(1.0, max(0.0, float(current_step) / float(total_steps)))
        target_b = min(float(max_batch_size), max(float(min_batch_size), b_noise * (0.5 + 0.5 * progress)))
        rec_batch = int(round(target_b))
        rec_batch = max(min_batch_size, min(max_batch_size, rec_batch))

        efficiency = 1.0 / (1.0 + (float(rec_batch) / max(1.0, b_noise)))

        return {
            "estimated_noise_scale_b_noise": b_noise,
            "gradient_trace_covariance": tr_cov,
            "mean_gradient_squared_norm": mean_g_sq_norm,
            "recommended_batch_size": rec_batch,
            "compute_efficiency": efficiency,
        }
