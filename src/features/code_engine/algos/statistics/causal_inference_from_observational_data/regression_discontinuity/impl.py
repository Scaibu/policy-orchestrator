from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoRegressionDiscontinuity:
    """
    ---
    contract:
      algo_id: ALGO-STAT-181
      name: StatsAlgoRegressionDiscontinuity
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - causal_inference_from_observational_data
        - regression_discontinuity
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
    Algorithm #181: Regression discontinuity
    Category: D7. Causal inference from observational data
    Definition: Estimating causal effects where treatment is assigned by a threshold on a score, by comparing units just above and just below the threshold.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Regression discontinuity.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Regression discontinuity (sharp RDD)
        cutoff = float(params.get("cutoff", 0.0))
        left_ctrl = [x for x in ctrl if x < cutoff]
        right_treat = [x for x in ctrl if x >= cutoff]
        m_left = sum(left_ctrl) / max(1, len(left_ctrl)) if left_ctrl else 0.0
        m_right = sum(right_treat) / max(1, len(right_treat)) if right_treat else 0.0
        jump = m_right - m_left
        results = {
            "cutoff": cutoff,
            "left_limit_mean": m_left,
            "right_limit_mean": m_right,
            "local_average_treatment_effect": jump
        }
    

        return {
            "status": "success",
            "results": results,
        }
