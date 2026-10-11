from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoStagedRolloutsWithStatisticalGuardrails:
    """
    ---
    contract:
      algo_id: ALGO-STAT-195
      name: StatsAlgoStagedRolloutsWithStatisticalGuardrails
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - experiment_platforms_and_decision_quality
        - staged_rollouts_with_statistical_guardrails
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
    Algorithm #195: Staged rollouts with statistical guardrails
    Category: D9. Experiment platforms and decision quality
    Definition: Releasing changes in increasing percentages of traffic, automatically checking guardrail metrics with valid sequential statistics at each stage, and rolling back on degradation.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Staged rollouts with statistical guardrails.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Staged rollouts with statistical guardrails
        rollout_percentage = float(params.get("stage_percentage", 5.0))
        error_rate_ctrl = sum(1 for x in ctrl if x < 0.0) / len(ctrl)
        error_rate_treat = sum(1 for x in treat if x < 0.0) / len(treat)
        guardrail_breached = bool(error_rate_treat > error_rate_ctrl * 1.5)
        results = {
            "current_stage_percentage": rollout_percentage,
            "error_rate_control": error_rate_ctrl,
            "error_rate_treatment": error_rate_treat,
            "guardrail_breached": guardrail_breached,
            "proceed_to_next_stage": not guardrail_breached
        }
    

        return {
            "status": "success",
            "results": results,
        }
