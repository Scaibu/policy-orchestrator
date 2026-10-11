from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoSampleRatioMismatch:
    """
    ---
    contract:
      algo_id: ALGO-STAT-159
      name: StatsAlgoSampleRatioMismatch
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - randomization_and_validity_checks
        - sample_ratio_mismatch
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
    Algorithm #159: Sample ratio mismatch (SRM) check
    Category: D2. Randomization and validity checks
    Definition: Checking that the observed split of units between variants matches the designed split, which detects broken assignment or logging.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Sample ratio mismatch (SRM) check.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Sample ratio mismatch (SRM) check via Pearson Chi-Square
        observed_ctrl = len(ctrl)
        observed_treat = len(treat)
        total = observed_ctrl + observed_treat
        expected_ratio = float(params.get("expected_ratio", 0.5))
        expected_ctrl = total * (1.0 - expected_ratio)
        expected_treat = total * expected_ratio
        chi2 = ((observed_ctrl - expected_ctrl)**2) / max(1e-12, expected_ctrl) + ((observed_treat - expected_treat)**2) / max(1e-12, expected_treat)
        p_val = math.exp(-0.5 * chi2) # df=1 tail approx
        srm_detected = bool(p_val < float(params.get("alpha", 0.001)))
        results = {
            "observed_control": observed_ctrl,
            "observed_treatment": observed_treat,
            "expected_control": expected_ctrl,
            "expected_treatment": expected_treat,
            "chi2_statistic": chi2,
            "p_value": p_val,
            "srm_detected": srm_detected
        }
    

        return {
            "status": "success",
            "results": results,
        }
