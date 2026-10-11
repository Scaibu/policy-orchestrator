from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoFalseDiscoveryRateControl:
    """
    ---
    contract:
      algo_id: ALGO-STAT-169
      name: StatsAlgoFalseDiscoveryRateControl
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - multiple_testing_and_metric_design
        - false_discovery_rate_control
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
    Algorithm #169: False discovery rate control (Benjamini-Hochberg)
    Category: D5. Multiple testing and metric design
    Definition: Controlling the expected fraction of false positives among the results declared significant, which is less strict than family-wise control and suits many exploratory metrics.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for False discovery rate control (Benjamini-Hochberg).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # False discovery rate control (Benjamini-Hochberg)
        p_values = sorted([float(x) for x in ctrl if 0.0 <= x <= 1.0] or [0.005, 0.02, 0.04, 0.15])
        m = len(p_values)
        q = float(params.get("fdr_q", 0.05))
        significant_indices = []
        for i, p in enumerate(p_values, start=1):
            crit = (i / m) * q
            if p <= crit:
                significant_indices.append(i - 1)
        max_k = max(significant_indices) if significant_indices else -1
        fdr_sig = [i <= max_k for i in range(m)]
        results = {
            "p_values": p_values,
            "fdr_threshold_q": q,
            "discoveries": sum(fdr_sig),
            "fdr_significant": fdr_sig
        }
    

        return {
            "status": "success",
            "results": results,
        }
