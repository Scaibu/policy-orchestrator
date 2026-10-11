from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoRandomizationAndHashBasedBucketing:
    """
    ---
    contract:
      algo_id: ALGO-STAT-158
      name: StatsAlgoRandomizationAndHashBasedBucketing
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - randomization_and_validity_checks
        - randomization_and_hash_based_bucketing
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
    Algorithm #158: Randomization and hash-based bucketing
    Category: D2. Randomization and validity checks
    Definition: Assigning units to experiment variants randomly but deterministically, so each unit always gets the same variant.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Randomization and hash-based bucketing.
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Randomization and hash-based bucketing
        import hashlib
        salt = str(params.get("salt", "exp_158_salt"))
        num_buckets = int(params.get("num_buckets", 100))
        assignments = []
        for i, val in enumerate(ctrl + treat):
            entity_id = f"user_{i}_{val}"
            hash_val = int(hashlib.md5(f"{salt}_{entity_id}".encode()).hexdigest(), 16)
            bucket = hash_val % num_buckets
            variant = "treatment" if bucket >= (num_buckets // 2) else "control"
            assignments.append({"id": entity_id, "bucket": bucket, "variant": variant})
        treat_count = sum(1 for a in assignments if a["variant"] == "treatment")
        ctrl_count = len(assignments) - treat_count
        results = {
            "total_units": len(assignments),
            "control_assigned": ctrl_count,
            "treatment_assigned": treat_count,
            "split_ratio": treat_count / max(1, len(assignments)),
            "bucket_sample": assignments[:5]
        }
    

        return {
            "status": "success",
            "results": results,
        }
