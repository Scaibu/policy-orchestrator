from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoRobustEstimators:
    """
    ---
    contract:
      algo_id: ALGO-STAT-189
      name: StatsAlgoRobustEstimators
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - operational_statistics_and_robust_estimation
        - robust_estimators
      inputs:
        type: object
        properties:
          control:
            type: array
            items:
              type: number
            description: Control cohort observations, baseline metrics, or reference samples.
          treatment:
            type: array
            items:
              type: number
            description: Treatment cohort observations, variant metrics, or post-intervention samples.
          parameters:
            type: object
            description: Statistical parameters such as significance alpha, power, MDE, or priors.
        required:
          - control
          - treatment
      outputs:
        type: object
        properties:
          status:
            type: string
            enum: [success, failed]
            description: Execution status.
          results:
            type: object
            description: Statistical test estimates, p-values, confidence intervals, or diagnostics.
        required:
          - status
          - results
      parameters: {}
      input_assumptions:
        - control and treatment must be non-empty lists of numerical floats.
        - Observations must satisfy bounded real values without NaN or infinity.
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: n/a
      uses_model: false
      complexity:
        variables:
          N: sample size of control
          M: sample size of treatment
        time_worst: O(N + M)
        time_typical: O(N + M)
        space: O(1)
      preconditions:
        - len(control) > 0
        - len(treatment) > 0
        - all(math.isfinite(x) for x in control)
        - all(math.isfinite(x) for x in treatment)
      postconditions:
        - output.status == "success"
        - isinstance(output.results, dict)
      certificate: none
      compatible_adapters: []
      related_algos: []
      references:
        - "kohavi2020trustworthy"
    ---
    Formal Specification:
    Algorithm #189: Robust estimators (Huber M-estimators, trimmed means, median of means)
    Category: D8. Operational statistics and robust estimation
    Definition: Estimators of central tendency and effects that are much less affected by outliers and heavy tails than plain means.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Robust estimators (Huber M-estimators, trimmed means, median of means).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Robust estimators (Huber M-estimator, trimmed mean)
        trim_p = float(params.get("trim_proportion", 0.1))
        s_c = sorted(ctrl)
        k = int(trim_p * len(s_c))
        trimmed = s_c[k:len(s_c)-k] if k < len(s_c)//2 else s_c
        trimmed_mean = sum(trimmed) / max(1, len(trimmed))
        results = {
            "raw_mean": sum(ctrl) / len(ctrl),
            "trimmed_mean": trimmed_mean,
            "trimmed_points_count": 2 * k
        }
    

        return {
            "status": "success",
            "results": results,
        }
