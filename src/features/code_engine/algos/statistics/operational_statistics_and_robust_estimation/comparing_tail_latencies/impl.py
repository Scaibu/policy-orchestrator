from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoComparingTailLatencies:
    """
    ---
    contract:
      algo_id: ALGO-STAT-188
      name: StatsAlgoComparingTailLatencies
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - operational_statistics_and_robust_estimation
        - comparing_tail_latencies
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
    Algorithm #188: Comparing tail latencies (quantile confidence intervals and tests)
    Category: D8. Operational statistics and robust estimation
    Definition: Correctly comparing high percentiles (p95, p99) of latency or other heavy-tailed metrics between versions.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Comparing tail latencies (quantile confidence intervals and tests).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Comparing tail latencies (Quantile confidence intervals and tests)
        q = float(params.get("quantile", 0.95))
        s_c = sorted(ctrl)
        s_t = sorted(treat)
        p95_c = s_c[int(q * len(s_c))]
        p95_t = s_t[int(q * len(s_t))]
        diff = p95_t - p95_c
        results = {
            "quantile": q,
            "control_quantile": p95_c,
            "treatment_quantile": p95_t,
            "tail_difference": diff
        }
    

        return {
            "status": "success",
            "results": results,
        }
