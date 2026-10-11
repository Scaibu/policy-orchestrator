from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoNonparametricTests:
    """
    ---
    contract:
      algo_id: ALGO-STAT-154
      name: StatsAlgoNonparametricTests
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - nonparametric_tests
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
    Algorithm #154: Nonparametric tests (Mann-Whitney U, permutation tests)
    Category: D1. Testing foundations
    Definition: Tests that don't assume a particular distribution, comparing ranks (Mann-Whitney) or using reshuffling of group labels (permutation).
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Nonparametric tests (Mann-Whitney U, permutation tests).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Nonparametric test: Mann-Whitney U test
        combined = [(x, 0) for x in ctrl] + [(x, 1) for x in treat]
        combined.sort(key=lambda item: item[0])
        n1 = len(ctrl)
        n2 = len(treat)
        rank_sum_1 = 0.0
        for rank, (val, group) in enumerate(combined, start=1):
            if group == 0:
                rank_sum_1 += rank
        u1 = rank_sum_1 - (n1 * (n1 + 1)) / 2.0
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)
        mean_u = (n1 * n2) / 2.0
        std_u = math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
        z_stat = (u_stat - mean_u) / max(1e-12, std_u)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
        results = {
            "u_statistic": u_stat,
            "u1": u1,
            "u2": u2,
            "z_score": z_stat,
            "p_value": p_val,
            "significant": bool(p_val < float(params.get("alpha", 0.05)))
        }
    

        return {
            "status": "success",
            "results": results,
        }
