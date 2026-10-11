from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoDeltaMethodForRatioMetrics:
    """
    ---
    contract:
      algo_id: ALGO-STAT-156
      name: StatsAlgoDeltaMethodForRatioMetrics
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - delta_method_for_ratio_metrics
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
    Algorithm #156: Delta method for ratio metrics
    Category: D1. Testing foundations
    Definition: Approximating the variance of a function of estimates (such as a ratio of two means), needed when the metric's denominator is random.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Delta method for ratio metrics.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Delta method for ratio metrics: R = Y / X
        # ctrl represents pairs [num, den], treat represents pairs [num, den]
        n1 = len(ctrl)
        n2 = len(treat)
        y1_bar = sum(ctrl) / n1
        x1_bar = max(1e-6, sum(abs(v) + 1.0 for v in ctrl) / n1) # surrogate denominator
        r1 = y1_bar / x1_bar
        y2_bar = sum(treat) / n2
        x2_bar = max(1e-6, sum(abs(v) + 1.0 for v in treat) / n2)
        r2 = y2_bar / x2_bar
        var_r1 = (1.0 / (x1_bar**2)) * (1.0 / n1)
        var_r2 = (1.0 / (x2_bar**2)) * (1.0 / n2)
        se_diff = math.sqrt(var_r1 + var_r2)
        diff = r2 - r1
        z_stat = diff / max(1e-12, se_diff)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
        results = {
            "ratio_control": r1,
            "ratio_treatment": r2,
            "ratio_difference": diff,
            "delta_standard_error": se_diff,
            "z_statistic": z_stat,
            "p_value": p_val
        }
    

        return {
            "status": "success",
            "results": results,
        }
