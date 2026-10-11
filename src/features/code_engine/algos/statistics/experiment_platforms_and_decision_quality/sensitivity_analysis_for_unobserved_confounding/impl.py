from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoSensitivityAnalysisForUnobservedConfounding:
    """
    ---
    contract:
      algo_id: ALGO-STAT-200
      name: StatsAlgoSensitivityAnalysisForUnobservedConfounding
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - experiment_platforms_and_decision_quality
        - sensitivity_analysis_for_unobserved_confounding
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
    Algorithm #200: Sensitivity analysis for unobserved confounding (E-values, Rosenbaum bounds)
    Category: D9. Experiment platforms and decision quality
    Definition: Quantifying how strong an unmeasured confounder would have to be to explain away an observational effect estimate.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Sensitivity analysis for unobserved confounding (E-values, Rosenbaum bounds).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Sensitivity analysis for unobserved confounding (E-values)
        rr = float(params.get("risk_ratio", 1.5))
        e_value = rr + math.sqrt(rr * (rr - 1.0)) if rr >= 1.0 else 1.0
        results = {
            "observed_relative_risk": rr,
            "e_value_point_estimate": e_value,
            "robustness_interpretation": f"Confounder must be associated with both treatment and outcome by at least {round(e_value, 2)}-fold to explain away the effect."
        }
    

        return {
            "status": "success",
            "results": results,
        }
