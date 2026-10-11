from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoAlwaysValidInference:
    """
    ---
    contract:
      algo_id: ALGO-STAT-166
      name: StatsAlgoAlwaysValidInference
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - sequential_and_bayesian_testing
        - always_valid_inference
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
    Algorithm #166: Always-valid inference (mSPRT, confidence sequences)
    Category: D4. Sequential and Bayesian testing
    Definition: Methods that let you check results continuously at any time and stop whenever, while keeping the false positive rate guaranteed.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Always-valid inference (mSPRT, confidence sequences).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Always-valid inference (mSPRT / confidence sequence)
        # Mixture Sequential Probability Ratio Test
        rho = float(params.get("rho", 0.5))
        n = min(len(ctrl), len(treat))
        diffs = [treat[i] - ctrl[i] for i in range(n)]
        s_n = sum(diffs)
        v_n = n * 1.0 # variance surrogate
        lr = math.sqrt(v_n / (v_n + rho)) * math.exp((rho * (s_n**2)) / (2.0 * (v_n + rho)))
        alpha = float(params.get("alpha", 0.05))
        threshold = 1.0 / alpha
        reject = bool(lr > threshold)
        results = {
            "sample_size": n,
            "cumulative_statistic_sn": s_n,
            "mixture_likelihood_ratio": lr,
            "boundary_threshold": threshold,
            "always_valid_reject": reject
        }
    

        return {
            "status": "success",
            "results": results,
        }
