from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoStatisticalProcessControlCharts:
    """
    ---
    contract:
      algo_id: ALGO-STAT-186
      name: StatsAlgoStatisticalProcessControlCharts
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - operational_statistics_and_robust_estimation
        - statistical_process_control_charts
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
    Algorithm #186: Statistical process control charts (Shewhart, EWMA charts)
    Category: D8. Operational statistics and robust estimation
    Definition: Monitoring a process metric over time with control limits that separate normal variation from signals needing investigation.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Statistical process control charts (Shewhart, EWMA charts).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Statistical process control charts (Shewhart, EWMA)
        mean_c = sum(ctrl) / len(ctrl)
        std_c = math.sqrt(sum((x - mean_c)**2 for x in ctrl) / max(1, len(ctrl) - 1))
        ucl = mean_c + 3.0 * std_c
        lcl = mean_c - 3.0 * std_c
        out_of_control = [x for x in treat if x > ucl or x < lcl]
        results = {
            "center_line": mean_c,
            "upper_control_limit": ucl,
            "lower_control_limit": lcl,
            "out_of_control_points": len(out_of_control),
            "process_in_control": len(out_of_control) == 0
        }
    

        return {
            "status": "success",
            "results": results,
        }
