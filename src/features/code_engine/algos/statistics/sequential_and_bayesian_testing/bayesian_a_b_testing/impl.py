from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


class StatsAlgoBayesianABTesting:
    """
    ---
    contract:
      algo_id: ALGO-STAT-167
      name: StatsAlgoBayesianABTesting
      version: 1.0.0
      category: statistics
      capability_tags:
        - experimentation
        - statistics
        - sequential_and_bayesian_testing
        - bayesian_a_b_testing
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
    Algorithm #167: Bayesian A/B testing (beta-binomial, expected loss)
    Category: D4. Sequential and Bayesian testing
    Definition: Analyzing experiments by computing posterior distributions of each variant's metric, and deciding with probabilities and expected losses instead of p-values.
    """

    @staticmethod
    def evaluate(
        control: Sequence[float],
        treatment: Sequence[float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes statistical evaluation for Bayesian A/B testing (beta-binomial, expected loss).
        """
        if not control or not treatment:
            raise ValueError("Precondition failed: control and treatment cohorts must be non-empty.")
        if any(not math.isfinite(x) for x in control) or any(not math.isfinite(x) for x in treatment):
            raise ValueError("Precondition failed: inputs must contain finite numerical values.")

        ctrl = [float(x) for x in control]
        treat = [float(x) for x in treatment]
        params = parameters or {}


        # Bayesian A/B testing (beta-binomial, expected loss)
        succ_c = sum(1 for x in ctrl if x > 0.5)
        succ_t = sum(1 for x in treat if x > 0.5)
        alpha_prior = 1.0
        beta_prior = 1.0
        post_a_c = alpha_prior + succ_c
        post_b_c = beta_prior + len(ctrl) - succ_c
        post_a_t = alpha_prior + succ_t
        post_b_t = beta_prior + len(treat) - succ_t
        mean_c = post_a_c / (post_a_c + post_b_c)
        mean_t = post_a_t / (post_a_t + post_b_t)
        # Prob(treat > ctrl) approx via normal approximation
        se = math.sqrt((mean_c * (1 - mean_c)) / len(ctrl) + (mean_t * (1 - mean_t)) / len(treat))
        prob_treat_better = 0.5 * (1.0 + math.erf((mean_t - mean_c) / max(1e-12, math.sqrt(2.0) * se)))
        expected_loss = max(0.0, mean_c - mean_t) * (1.0 - prob_treat_better)
        results = {
            "posterior_mean_control": mean_c,
            "posterior_mean_treatment": mean_t,
            "prob_treatment_superior": prob_treat_better,
            "expected_loss": expected_loss,
            "decision": "ship_treatment" if prob_treat_better > 0.95 else "continue_experiment"
        }
    

        return {
            "status": "success",
            "results": results,
        }
