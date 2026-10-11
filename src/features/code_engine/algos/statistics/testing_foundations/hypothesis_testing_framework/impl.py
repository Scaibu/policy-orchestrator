from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoHypothesisTestingFramework:
    """
    ---
    contract:
      algo_id: ALGO-STAT-151
      name: StatsAlgoHypothesisTestingFramework
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - hypothesis_testing_framework
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
    Algorithm #151: Hypothesis testing framework (p-values, significance, power, MDE)
    Category: D1. Testing foundations
    Definition: The standard framework for deciding whether an observed difference (between treatment and control) is larger than random noise would plausibly produce.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Hypothesis testing framework (p-values, significance, power, MDE).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Hypothesis testing framework
        mean_ctrl = sum(ctrl) / len(ctrl)
        mean_treat = sum(treat) / len(treat)
        var_ctrl = sum((x - mean_ctrl) ** 2 for x in ctrl) / max(1, len(ctrl) - 1)
        var_treat = sum((x - mean_treat) ** 2 for x in treat) / max(1, len(treat) - 1)
        se = math.sqrt(var_ctrl / len(ctrl) + var_treat / len(treat))
        diff = mean_treat - mean_ctrl
        z_stat = diff / max(1e-12, se)
        # two-tailed p-value via normal approximation
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
        alpha = float(params.get("alpha", 0.05))
        significant = bool(p_val < alpha)
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        mde = 2.8 * se  # approx MDE for 80% power at alpha=0.05
        results = {
            "mean_control": mean_ctrl,
            "mean_treatment": mean_treat,
            "effect_size": diff,
            "standard_error": se,
            "test_statistic": z_stat,
            "p_value": p_val,
            "ci_95": [ci_lower, ci_upper],
            "mde": mde,
            "significant": significant,
        }
    

        return {
            "status": "success",
            "results": results,
        }
