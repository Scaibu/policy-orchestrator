from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoGenerativeAdversarialNetwork:
    """
    ---
    contract:
      algo_id: ALGO-NN-154
      name: NnAlgoGenerativeAdversarialNetwork
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - gan
        - min_max_game
        - adversarial_training
      inputs:
        type: object
        required:
          - real_probabilities
          - fake_probabilities
        properties:
          real_probabilities:
            type: array
            items:
              type: number
            description: Discriminator predicted probability D(x) for real samples, in (0, 1).
          fake_probabilities:
            type: array
            items:
              type: number
            description: Discriminator predicted probability D(G(z)) for fake samples, in (0, 1).
          label_smoothing:
            type: number
            default: 0.0
            description: One-sided real label smoothing parameter alpha.
      outputs:
        type: object
        required:
          - discriminator_loss
          - generator_loss
          - d_real_accuracy
          - d_fake_accuracy
        properties:
          discriminator_loss:
            type: number
            description: Binary cross-entropy discriminator objective.
          generator_loss:
            type: number
            description: Non-saturating generator minimax objective -log(D(G(z))).
          d_real_accuracy:
            type: number
            description: Classification accuracy on real batch.
          d_fake_accuracy:
            type: number
            description: Classification accuracy on fake batch.
      parameters: {}
      input_assumptions:
        - real_probabilities and fake_probabilities are non-empty
        - probabilities lie strictly in [0.0, 1.0]
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically clamped with epsilon 1e-12 to prevent log(0)"
      uses_model: false
      complexity:
        variables:
          N_real: number of real samples
          N_fake: number of generated samples
        time_worst: O(N_real + N_fake)
        time_typical: O(N_real + N_fake)
        space: O(1)
      preconditions:
        - len(input.real_probabilities) > 0 and len(input.fake_probabilities) > 0
        - all(0.0 <= p <= 1.0 for p in input.real_probabilities)
        - all(0.0 <= p <= 1.0 for p in input.fake_probabilities)
        - 0.0 <= input.label_smoothing < 0.5
      postconditions:
        - output.discriminator_loss >= 0.0
        - output.generator_loss >= 0.0
        - 0.0 <= output.d_real_accuracy <= 1.0
        - 0.0 <= output.d_fake_accuracy <= 1.0
      certificate: "Discriminator loss is non-negative cross-entropy"
      compatible_adapters:
        - ADAPTER-ADVERSARIAL-TRAINER
      related_algos:
        - ALGO-NN-155
        - ALGO-NN-156
      references:
        - "https://arxiv.org/abs/1406.2661"
    ---
    """

    @staticmethod
    def forward(
        real_probabilities: List[float],
        fake_probabilities: List[float],
        label_smoothing: float = 0.0,
    ) -> Dict[str, Any]:
        N_r = len(real_probabilities)
        N_f = len(fake_probabilities)
        if N_r == 0 or N_f == 0:
            raise ValueError("Precondition failed: probability lists cannot be empty")

        eps = 1e-12
        target_real = 1.0 - label_smoothing

        # Discriminator real loss: - (y_real * log(p) + (1-y_real) * log(1-p))
        d_real_loss_sum = 0.0
        real_correct = 0
        for p in real_probabilities:
            clamped_p = max(eps, min(1.0 - eps, p))
            loss = -(target_real * math.log(clamped_p) + (1.0 - target_real) * math.log(1.0 - clamped_p))
            d_real_loss_sum += loss
            if p >= 0.5:
                real_correct += 1

        # Discriminator fake loss: - log(1 - p_fake)
        d_fake_loss_sum = 0.0
        fake_correct = 0
        g_loss_sum = 0.0
        for p in fake_probabilities:
            clamped_p = max(eps, min(1.0 - eps, p))
            d_fake_loss_sum += -math.log(1.0 - clamped_p)
            # Non-saturating generator loss: - log(p_fake)
            g_loss_sum += -math.log(clamped_p)
            if p < 0.5:
                fake_correct += 1

        d_loss = (d_real_loss_sum / float(N_r) + d_fake_loss_sum / float(N_f)) * 0.5
        g_loss = g_loss_sum / float(N_f)

        return {
            "discriminator_loss": d_loss,
            "generator_loss": g_loss,
            "d_real_accuracy": float(real_correct) / float(N_r),
            "d_fake_accuracy": float(fake_correct) / float(N_f),
        }
