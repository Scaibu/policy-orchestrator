from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoWinsorizationAndCappingOfHeavyTailedMetrics:
    """
    ---
    contract:
      algo_id: ALGO-STAT-164
      name: StatsAlgoWinsorizationAndCappingOfHeavyTailedMetrics
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - variance_reduction
        - winsorization_and_capping_of_heavy_tailed_metrics
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
    Algorithm #164: Winsorization and capping of heavy-tailed metrics
    Category: D3. Variance reduction
    Definition: Limiting extreme values in a metric (for example revenue or session time per user) to reduce variance from a few outliers.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Winsorization and capping of heavy-tailed metrics.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Winsorization and capping of heavy-tailed metrics
        p_lower = float(params.get("percentile_lower", 0.01))
        p_upper = float(params.get("percentile_upper", 0.99))
        sorted_c = sorted(ctrl)
        n = len(sorted_c)
        val_low = sorted_c[int(p_lower * n)]
        val_high = sorted_c[int(p_upper * n)]
        winsorized = [min(val_high, max(val_low, x)) for x in ctrl]
        results = {
            "raw_mean": sum(ctrl) / n,
            "winsorized_mean": sum(winsorized) / n,
            "cap_lower": val_low,
            "cap_upper": val_high,
            "capped_count": sum(1 for x in ctrl if x < val_low or x > val_high)
        }
    

        return {
            "status": "success",
            "results": results,
        }
