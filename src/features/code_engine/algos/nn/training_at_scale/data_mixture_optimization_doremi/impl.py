from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDataMixtureOptimizationDoremi:
    """
    ---
    contract:
      algo_id: ALGO-NN-243
      name: NnAlgoDataMixtureOptimizationDoremi
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - data_mixture
        - doremi
        - distributionally_robust_optimization
        - excess_loss
        - domain_reweighting
      inputs:
        type: object
        required:
          - domain_losses_proxy
          - domain_losses_reference
          - current_domain_weights
        properties:
          domain_losses_proxy:
            type: array
            items:
              type: number
            description: Empirical cross-entropy losses evaluated on small proxy model across D domains.
          domain_losses_reference:
            type: array
            items:
              type: number
            description: Baseline losses evaluated on pre-trained reference model across D domains.
          current_domain_weights:
            type: array
            items:
              type: number
            description: Current sampling mixture distribution alphas across D domains (summing to 1).
          step_size_eta:
            type: number
            default: 0.1
            description: Exponent step size hyperparameter for mirror descent weight update.
          smoothing_epsilon:
            type: number
            default: 0.001
            description: Minimum probability floor to prevent domain starvation.
      outputs:
        type: object
        required:
          - updated_domain_weights
          - excess_losses
          - worst_case_domain_idx
          - max_excess_loss
          - num_domains
        properties:
          updated_domain_weights:
            type: array
            items:
              type: number
            description: Optimized sampling mixture probabilities for subsequent pre-training stages.
          excess_losses:
            type: array
            items:
              type: number
            description: Domain excess losses: L_proxy,d - L_reference,d.
          worst_case_domain_idx:
            type: integer
            description: Domain experiencing largest excess learning deficit.
          max_excess_loss:
            type: number
            description: Magnitude of the maximum excess loss.
          num_domains:
            type: integer
            description: Total number of data domains D.
      complexity:
        time: O(D)
        space: O(D)
      preconditions:
        - len(input.domain_losses_proxy) > 0
        - len(input.domain_losses_proxy) == len(input.domain_losses_reference)
        - len(input.domain_losses_proxy) == len(input.current_domain_weights)
        - all(w >= 0.0 for w in input.current_domain_weights)
        - input.step_size_eta > 0.0
      postconditions:
        - len(output.updated_domain_weights) == len(input.domain_losses_proxy)
        - abs(sum(output.updated_domain_weights) - 1.0) < 1e-5
        - all(w > 0.0 for w in output.updated_domain_weights)
      certificate: "Domain weights are updated via exponentiated excess loss and normalized to valid simplex"
      compatible_adapters:
        - ADAPTER-DOREMI-MIXTURE-OPTIMIZER
        - ADAPTER-DOMAIN-ROBUST-SAMPLER
      related_algos:
        - ALGO-NN-207
        - ALGO-NN-245
      references:
        - "https://arxiv.org/abs/2305.10429"
        - "https://arxiv.org/abs/2406.17562"
    ---
    """

    @classmethod
    def forward(
        cls,
        domain_losses_proxy: List[float],
        domain_losses_reference: List[float],
        current_domain_weights: List[float],
        step_size_eta: float = 0.1,
        smoothing_epsilon: float = 0.001,
    ) -> Dict[str, Any]:
        num_domains = len(domain_losses_proxy)
        if num_domains == 0:
            raise ValueError("Precondition failed: domain_losses_proxy cannot be empty")
        if len(domain_losses_reference) != num_domains or len(current_domain_weights) != num_domains:
            raise ValueError("Precondition failed: vector dimensions across domains must match")
        if any(w < 0.0 for w in current_domain_weights):
            raise ValueError("Precondition failed: weights must be non-negative")
        if step_size_eta <= 0.0:
            raise ValueError("Precondition failed: step_size_eta must be positive")

        # 1. Compute excess loss: delta_d = L_proxy,d - L_reference,d
        excess_losses: List[float] = []
        for d in range(num_domains):
            excess_losses.append(domain_losses_proxy[d] - domain_losses_reference[d])

        worst_idx = excess_losses.index(max(excess_losses))
        max_excess = excess_losses[worst_idx]

        # 2. Exponentiated gradient / mirror descent update:
        # alpha_new,d \propto alpha_old,d * exp(eta * delta_d)
        unnormalized: List[float] = []
        for d in range(num_domains):
            w_old = max(1e-12, current_domain_weights[d])
            delta = excess_losses[d]
            # Clip exponent argument for numerical safety
            scaled_delta = min(20.0, max(-20.0, step_size_eta * delta))
            unnormalized.append(w_old * math.exp(scaled_delta))

        sum_unnorm = sum(unnormalized)
        inv_sum = 1.0 / max(1e-12, sum_unnorm)

        # 3. Add epsilon smoothing floor and renormalize
        smoothed: List[float] = []
        for d in range(num_domains):
            smoothed.append((unnormalized[d] * inv_sum) + smoothing_epsilon)

        sum_smoothed = sum(smoothed)
        updated_weights = [w / sum_smoothed for w in smoothed]

        return {
            "updated_domain_weights": updated_weights,
            "excess_losses": excess_losses,
            "worst_case_domain_idx": worst_idx,
            "max_excess_loss": max_excess,
            "num_domains": num_domains,
        }
