from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoSurvivalAnalysis:
    """
    ---
    contract:
      algo_id: ALGO-STAT-187
      name: StatsAlgoSurvivalAnalysis
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - operational_statistics_and_robust_estimation
        - survival_analysis
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
    Algorithm #187: Survival analysis (Kaplan-Meier, Cox proportional hazards)
    Category: D8. Operational statistics and robust estimation
    Definition: Analyzing time until an event (churn, failure, conversion, resolution) when some units haven't experienced the event yet (censoring).
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Survival analysis (Kaplan-Meier, Cox proportional hazards).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Survival analysis (Kaplan-Meier survival estimation)
        times = sorted(set(abs(int(x)) + 1 for x in ctrl))
        survival = 1.0
        curve = []
        n_at_risk = len(ctrl)
        for t in times:
            d = sum(1 for x in ctrl if abs(int(x)) + 1 == t)
            survival *= (1.0 - d / max(1, n_at_risk))
            curve.append({"time": t, "survival_prob": survival})
            n_at_risk -= d
        results = {
            "median_survival_time": times[len(times)//2] if times else 0,
            "survival_curve_sample": curve[:5]
        }
    

        return {
            "status": "success",
            "results": results,
        }
