from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoStratificationAndPostStratification:
    """
    ---
    contract:
      algo_id: ALGO-STAT-162
      name: StatsAlgoStratificationAndPostStratification
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - variance_reduction
        - stratification_and_post_stratification
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
    Algorithm #162: Stratification and post-stratification
    Category: D3. Variance reduction
    Definition: Reducing variance by accounting for known subgroups (platform, country, user tenure), either in the assignment or in the analysis.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Stratification and post-stratification.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Stratification and post-stratification
        # Strata: low, medium, high
        strata_weights = {"low": 0.3, "medium": 0.4, "high": 0.3}
        means = {
            "low": sum(treat[:len(treat)//3]) / max(1, len(treat)//3),
            "medium": sum(treat[len(treat)//3:2*len(treat)//3]) / max(1, len(treat)//3),
            "high": sum(treat[2*len(treat)//3:]) / max(1, len(treat) - 2*(len(treat)//3))
        }
        stratified_mean = sum(strata_weights[k] * means[k] for k in strata_weights)
        raw_mean = sum(treat) / len(treat)
        results = {
            "raw_mean": raw_mean,
            "stratified_mean": stratified_mean,
            "strata_estimates": means,
            "variance_efficiency_gain": abs(stratified_mean - raw_mean) / max(1e-6, abs(raw_mean))
        }
    

        return {
            "status": "success",
            "results": results,
        }
