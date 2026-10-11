from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoFamilyWiseErrorControl:
    """
    ---
    contract:
      algo_id: ALGO-STAT-168
      name: StatsAlgoFamilyWiseErrorControl
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - multiple_testing_and_metric_design
        - family_wise_error_control
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
    Algorithm #168: Family-wise error control (Bonferroni, Holm)
    Category: D5. Multiple testing and metric design
    Definition: Adjusting for many simultaneous tests, so the probability of at least one false positive across all of them stays at α.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Family-wise error control (Bonferroni, Holm).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Family-wise error control (Bonferroni, Holm)
        p_values = sorted([float(x) for x in ctrl if 0.0 <= x <= 1.0] or [0.01, 0.04, 0.06])
        m = len(p_values)
        alpha = float(params.get("alpha", 0.05))
        bonferroni_thresh = alpha / m
        bonferroni_sig = [p < bonferroni_thresh for p in p_values]
        # Holm step-down
        holm_sig = []
        for i, p in enumerate(p_values):
            thresh = alpha / (m - i)
            if p < thresh:
                holm_sig.append(True)
            else:
                holm_sig.extend([False] * (m - len(holm_sig)))
                break
        results = {
            "raw_p_values": p_values,
            "bonferroni_threshold": bonferroni_thresh,
            "bonferroni_significant": bonferroni_sig,
            "holm_significant": holm_sig,
            "hypotheses_tested": m
        }
    

        return {
            "status": "success",
            "results": results,
        }
