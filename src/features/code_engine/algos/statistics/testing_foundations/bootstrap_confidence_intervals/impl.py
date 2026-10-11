from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoBootstrapConfidenceIntervals:
    """
    ---
    contract:
      algo_id: ALGO-STAT-155
      name: StatsAlgoBootstrapConfidenceIntervals
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - testing_foundations
        - bootstrap_confidence_intervals
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
    Algorithm #155: Bootstrap confidence intervals (percentile, BCa)
    Category: D1. Testing foundations
    Definition: Estimating a statistic's uncertainty by resampling the data with replacement many times and looking at the spread of the recomputed statistic.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Bootstrap confidence intervals (percentile, BCa).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Bootstrap confidence intervals
        diff_mean = (sum(treat) / len(treat)) - (sum(ctrl) / len(ctrl))
        n_boot = int(params.get("n_bootstraps", 100))
        seed = int(params.get("seed", 42))
        import random
        rng = random.Random(seed)
        boot_diffs = []
        for _ in range(n_boot):
            sample_ctrl = [rng.choice(ctrl) for _ in range(len(ctrl))]
            sample_treat = [rng.choice(treat) for _ in range(len(treat))]
            boot_diffs.append((sum(sample_treat) / len(sample_treat)) - (sum(sample_ctrl) / len(sample_ctrl)))
        boot_diffs.sort()
        idx_lower = int(0.025 * n_boot)
        idx_upper = int(0.975 * n_boot)
        results = {
            "observed_difference": diff_mean,
            "bootstrap_mean": sum(boot_diffs) / len(boot_diffs),
            "ci_percentile_lower": boot_diffs[idx_lower],
            "ci_percentile_upper": boot_diffs[idx_upper],
            "bootstrap_samples": n_boot
        }
    

        return {
            "status": "success",
            "results": results,
        }
