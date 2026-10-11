from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoTestsForProportions:
    """
    ---
    contract:
      algo_id: ALGO-STAT-153
      name: StatsAlgoTestsForProportions
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - tests_for_proportions
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
    Algorithm #153: Tests for proportions (z-test, chi-squared, Fisher's exact test)
    Category: D1. Testing foundations
    Definition: Tests comparing rates (conversion, error rate, click-through) between groups.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Tests for proportions (z-test, chi-squared, Fisher's exact test).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Tests for proportions (two-proportion z-test)
        n1 = len(ctrl)
        n2 = len(treat)
        x1 = sum(1.0 for x in ctrl if x > 0.5)
        x2 = sum(1.0 for x in treat if x > 0.5)
        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
        z_stat = (p2 - p1) / max(1e-12, se)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
        results = {
            "rate_control": p1,
            "rate_treatment": p2,
            "rate_difference": p2 - p1,
            "pooled_rate": p_pool,
            "z_statistic": z_stat,
            "p_value": p_val,
            "significant": bool(p_val < float(params.get("alpha", 0.05)))
        }
    

        return {
            "status": "success",
            "results": results,
        }
