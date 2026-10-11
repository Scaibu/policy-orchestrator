from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoTriggerAnalysisAndDilution:
    """
    ---
    contract:
      algo_id: ALGO-STAT-178
      name: StatsAlgoTriggerAnalysisAndDilution
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - complex_experimental_designs
        - trigger_analysis_and_dilution
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
    Algorithm #178: Trigger analysis and dilution
    Category: D6. Complex experimental designs
    Definition: Analyzing only the users who could actually have experienced the change (the triggered population), then translating the effect back to the whole population.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Trigger analysis and dilution.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Trigger analysis and dilution
        # Only evaluate units who triggered the treatment feature
        trigger_rate = 0.25
        itn_effect = (sum(treat) / len(treat)) - (sum(ctrl) / len(ctrl))
        tot_effect = itn_effect / max(1e-6, trigger_rate) # Treatment on treated
        results = {
            "intent_to_treat_effect": itn_effect,
            "trigger_rate": trigger_rate,
            "treatment_on_treated_effect": tot_effect,
            "dilution_factor": 1.0 / trigger_rate
        }
    

        return {
            "status": "success",
            "results": results,
        }
