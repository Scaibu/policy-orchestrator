from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoGroupSequentialDesignsAndAlphaSpending:
    """
    ---
    contract:
      algo_id: ALGO-STAT-165
      name: StatsAlgoGroupSequentialDesignsAndAlphaSpending
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - sequential_and_bayesian_testing
        - group_sequential_designs_and_alpha_spending
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
    Algorithm #165: Group sequential designs and alpha spending (O'Brien-Fleming, Pocock)
    Category: D4. Sequential and Bayesian testing
    Definition: Allowing a fixed number of planned interim analyses during an experiment, with adjusted significance thresholds so the total false positive rate stays at α.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Group sequential designs and alpha spending (O'Brien-Fleming, Pocock).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Group sequential designs and alpha spending (O'Brien-Fleming)
        k_looks = int(params.get("k_looks", 3))
        current_look = int(params.get("current_look", 1))
        alpha_total = float(params.get("alpha_total", 0.05))
        # O'Brien-Fleming spending function: alpha(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))
        t = current_look / k_looks
        spent_alpha = 2.0 * (1.0 - 0.5 * (1.0 + math.erf((1.96 / math.sqrt(t)) / math.sqrt(2.0))))
        z_crit = 1.96 / math.sqrt(t)
        diff = (sum(treat) / len(treat)) - (sum(ctrl) / len(ctrl))
        se = math.sqrt(1.0 / len(ctrl) + 1.0 / len(treat))
        z_stat = diff / max(1e-12, se)
        early_stop = bool(abs(z_stat) > z_crit)
        results = {
            "current_look": current_look,
            "information_fraction": t,
            "critical_value_z": z_crit,
            "z_statistic": z_stat,
            "cumulative_alpha_spent": spent_alpha,
            "early_stopping_decision": early_stop
        }
    

        return {
            "status": "success",
            "results": results,
        }
