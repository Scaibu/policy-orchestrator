from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMixtureDensityNetworks:
    """
    ---
    contract:
      algo_id: ALGO-NN-188
      name: NnAlgoMixtureDensityNetworks
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - mixture_density_networks
        - mdn
        - multimodal_regression
        - gaussian_mixture
      inputs:
        type: object
        required:
          - mixture_logits
          - component_means
          - component_log_stds
          - target_y
        properties:
          mixture_logits:
            type: array
            items:
              type: number
            description: Unnormalized mixture coefficient logits alpha_k of length K_components.
          component_means:
            type: array
            items:
              type: number
            description: Gaussian component means mu_k of length K_components.
          component_log_stds:
            type: array
            items:
              type: number
            description: Log standard deviations log(sigma_k) of length K_components.
          target_y:
            type: number
            description: Scalar continuous observation target y.
      outputs:
        type: object
        required:
          - negative_log_likelihood
          - mixture_probabilities
          - expected_mean
          - total_predictive_variance
          - dominant_mode_mean
        properties:
          negative_log_likelihood:
            type: number
            description: Exact NLL loss -log sum_k pi_k * N(y; mu_k, sigma_k^2).
          mixture_probabilities:
            type: array
            items:
              type: number
            description: Softmax mixture component weights pi_k of length K_components.
          expected_mean:
            type: number
            description: Law of total expectation mean E[y] = sum_k pi_k * mu_k.
          total_predictive_variance:
            type: number
            description: Law of total variance Var[y] = sum_k pi_k * (sigma_k^2 + mu_k^2) - E[y]^2.
          dominant_mode_mean:
            type: number
            description: Mean of the Gaussian component with highest mixture weight pi_k.
      parameters: {}
      input_assumptions:
        - mixture_logits, component_means, and component_log_stds have matching length K >= 1
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact log-sum-exp stabilization"
      uses_model: false
      complexity:
        variables:
          K: number of mixture components
        time_worst: O(K)
        time_typical: O(K)
        space: O(K)
      preconditions:
        - len(input.mixture_logits) > 0
        - len(input.mixture_logits) == len(input.component_means) == len(input.component_log_stds)
      postconditions:
        - len(output.mixture_probabilities) == len(input.mixture_logits)
        - output.total_predictive_variance >= 0.0
      certificate: "Predictive variance decomposes strictly into epistemic and aleatoric variance"
      compatible_adapters:
        - ADAPTER-MDN-MULTIMODAL-HEAD
        - ADAPTER-UNCERTAINTY-REGRESSOR
      related_algos:
        - ALGO-NN-7
        - ALGO-NN-152
      references:
        - "https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf"
    ---
    """

    @classmethod
    def forward(
        cls,
        mixture_logits: List[float],
        component_means: List[float],
        component_log_stds: List[float],
        target_y: float,
    ) -> Dict[str, Any]:
        K = len(mixture_logits)
        if K == 0:
            raise ValueError("Precondition failed: at least one mixture component required")
        if len(component_means) != K or len(component_log_stds) != K:
            raise ValueError("Precondition failed: component parameter dimensions must match")

        # 1. Softmax mixture weights pi_k = exp(alpha_k) / sum exp(alpha)
        max_alpha = max(mixture_logits)
        exps = [math.exp(a - max_alpha) for a in mixture_logits]
        sum_exp = sum(exps)
        pi_probs = [e / sum_exp for e in exps]
        log_pi = [a - max_alpha - math.log(sum_exp) for a in mixture_logits]

        # 2. Compute component log-likelihoods:
        # log N(y; mu_k, sigma_k^2) = -0.5 * log(2 * pi) - log_sigma_k - 0.5 * ((y - mu_k) / sigma_k)^2
        half_log_2pi = 0.5 * math.log(2.0 * math.pi)
        log_terms: List[float] = []

        for k in range(K):
            mu_k = component_means[k]
            log_sig_k = component_log_stds[k]
            sig_k = math.exp(log_sig_k)

            z = (target_y - mu_k) / sig_k
            log_prob_k = -half_log_2pi - log_sig_k - 0.5 * z * z
            log_terms.append(log_pi[k] + log_prob_k)

        # 3. Log-Sum-Exp to find total log-likelihood:
        # log p(y) = LSE_k (log_pi_k + log N_k)
        max_log_term = max(log_terms)
        lse = max_log_term + math.log(sum(math.exp(lt - max_log_term) for lt in log_terms))
        nll = -lse

        # 4. Law of Total Expectation and Total Variance:
        # E[y] = sum_k pi_k * mu_k
        # Var[y] = sum_k pi_k * (sigma_k^2 + mu_k^2) - E[y]^2
        expected_mean = sum(pi_probs[k] * component_means[k] for k in range(K))
        expected_second_moment = sum(
            pi_probs[k] * (math.exp(2.0 * component_log_stds[k]) + component_means[k] ** 2)
            for k in range(K)
        )
        predictive_variance = max(0.0, expected_second_moment - expected_mean * expected_mean)

        best_k = max(range(K), key=lambda idx: pi_probs[idx])
        dominant_mode = component_means[best_k]

        return {
            "negative_log_likelihood": nll,
            "mixture_probabilities": pi_probs,
            "expected_mean": expected_mean,
            "total_predictive_variance": predictive_variance,
            "dominant_mode_mean": dominant_mode,
        }
