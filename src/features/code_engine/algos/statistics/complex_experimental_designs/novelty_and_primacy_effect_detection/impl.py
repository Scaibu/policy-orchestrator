from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoNoveltyAndPrimacyEffectDetection:
    """
    ---
    contract:
      algo_id: ALGO-STAT-177
      name: StatsAlgoNoveltyAndPrimacyEffectDetection
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - complex_experimental_designs
        - novelty_and_primacy_effect_detection
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
    Algorithm #177: Novelty and primacy effect detection
    Category: D6. Complex experimental designs
    Definition: Detecting effects that change over time because users react to the newness of a change (novelty: initial excitement fades; primacy: initial resistance fades).
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Novelty and primacy effect detection.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Novelty and primacy effect detection
        # Decay of effect over time segments
        n_periods = 4
        k = len(treat) // n_periods
        period_effects = []
        for p in range(n_periods):
            m_t = sum(treat[p*k:(p+1)*k]) / k
            m_c = sum(ctrl[p*k:(p+1)*k]) / k
            period_effects.append(m_t - m_c)
        novelty_detected = bool(period_effects[0] > 1.5 * period_effects[-1])
        results = {
            "period_effects": period_effects,
            "initial_effect": period_effects[0],
            "steady_state_effect": period_effects[-1],
            "novelty_effect_detected": novelty_detected
        }
    

        return {
            "status": "success",
            "results": results,
        }
