from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoTTestsAndWelchSTTest:
    """
    ---
    contract:
      algo_id: ALGO-STAT-152
      name: StatsAlgoTTestsAndWelchSTTest
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - t_tests_and_welch_s_t_test
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
    Algorithm #152: t-tests and Welch's t-test
    Category: D1. Testing foundations
    Definition: Tests comparing the means of two groups, with Welch's version not assuming equal variances.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for t-tests and Welch's t-test.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Welch's t-test
        n1 = len(ctrl)
        n2 = len(treat)
        m1 = sum(ctrl) / n1
        m2 = sum(treat) / n2
        s1_sq = sum((x - m1)**2 for x in ctrl) / max(1, n1 - 1)
        s2_sq = sum((x - m2)**2 for x in treat) / max(1, n2 - 1)
        se = math.sqrt(s1_sq / n1 + s2_sq / n2)
        diff = m2 - m1
        t_stat = diff / max(1e-12, se)
        # Welch-Satterthwaite degrees of freedom
        num_df = (s1_sq / n1 + s2_sq / n2)**2
        den_df = ((s1_sq / n1)**2) / max(1, n1 - 1) + ((s2_sq / n2)**2) / max(1, n2 - 1)
        dof = num_df / max(1e-12, den_df)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
        results = {
            "t_statistic": t_stat,
            "degrees_of_freedom": dof,
            "mean_diff": diff,
            "standard_error": se,
            "p_value": p_val,
            "significant": bool(p_val < float(params.get("alpha", 0.05)))
        }
    

        return {
            "status": "success",
            "results": results,
        }
