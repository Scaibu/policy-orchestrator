from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoConditionalImageToImageGan:
    """
    ---
    contract:
      algo_id: ALGO-NN-157
      name: NnAlgoConditionalImageToImageGan
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - cyclegan
        - pix2pix
        - conditional_gan
        - image_to_image
      inputs:
        type: object
        required:
          - real_source_a
          - real_target_b
          - fake_target_b
          - cycle_recon_a
          - fake_source_a
          - cycle_recon_b
        properties:
          real_source_a:
            type: array
            items:
              type: number
            description: Real sample vector from domain A of dimension d_a.
          real_target_b:
            type: array
            items:
              type: number
            description: Real sample vector from domain B of dimension d_b.
          fake_target_b:
            type: array
            items:
              type: number
            description: Translated sample G(A) in domain B of dimension d_b.
          cycle_recon_a:
            type: array
            items:
              type: number
            description: Cycle-reconstructed sample F(G(A)) in domain A of dimension d_a.
          fake_source_a:
            type: array
            items:
              type: number
            description: Translated sample F(B) in domain A of dimension d_a.
          cycle_recon_b:
            type: array
            items:
              type: number
            description: Cycle-reconstructed sample G(F(B)) in domain B of dimension d_b.
          lambda_cycle:
            type: number
            default: 10.0
            description: Weight for cycle consistency loss.
          lambda_l1:
            type: number
            default: 100.0
            description: Weight for paired L1 loss (pix2pix mode).
          is_paired:
            type: boolean
            default: false
            description: Flag indicating paired supervised ground truth available.
      outputs:
        type: object
        required:
          - cycle_consistency_loss
          - l1_paired_loss
          - total_generator_objective
        properties:
          cycle_consistency_loss:
            type: number
            description: Mean L1 cycle reconstruction error across both directions.
          l1_paired_loss:
            type: number
            description: Supervised L1 error between G(A) and paired target B.
          total_generator_objective:
            type: number
            description: Weighted composite structural objective.
      parameters: {}
      input_assumptions:
        - real_source_a, cycle_recon_a, and fake_source_a have matching dimension d_a
        - real_target_b, fake_target_b, and cycle_recon_b have matching dimension d_b
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact analytical computation within IEEE-754 precision"
      uses_model: false
      complexity:
        variables:
          d_a: domain A feature dimension
          d_b: domain B feature dimension
        time_worst: O(d_a + d_b)
        time_typical: O(d_a + d_b)
        space: O(1)
      preconditions:
        - len(input.real_source_a) > 0 and len(input.real_source_a) == len(input.cycle_recon_a) == len(input.fake_source_a)
        - len(input.real_target_b) > 0 and len(input.real_target_b) == len(input.fake_target_b) == len(input.cycle_recon_b)
        - input.lambda_cycle >= 0.0
        - input.lambda_l1 >= 0.0
      postconditions:
        - output.cycle_consistency_loss >= 0.0
        - output.l1_paired_loss >= 0.0
        - output.total_generator_objective >= 0.0
      certificate: "Total objective equals lambda_cycle * cycle_loss + (lambda_l1 * l1_loss if paired else 0)"
      compatible_adapters:
        - ADAPTER-DOMAIN-TRANSLATOR
      related_algos:
        - ALGO-NN-154
        - ALGO-NN-155
      references:
        - "https://arxiv.org/abs/1703.10593"
        - "https://arxiv.org/abs/1611.07004"
    ---
    """

    @staticmethod
    def forward(
        real_source_a: List[float],
        real_target_b: List[float],
        fake_target_b: List[float],
        cycle_recon_a: List[float],
        fake_source_a: List[float],
        cycle_recon_b: List[float],
        lambda_cycle: float = 10.0,
        lambda_l1: float = 100.0,
        is_paired: bool = False,
    ) -> Dict[str, Any]:
        d_a = len(real_source_a)
        d_b = len(real_target_b)
        if d_a == 0 or d_b == 0:
            raise ValueError("Precondition failed: feature dimensions must be non-zero")

        if len(cycle_recon_a) != d_a or len(fake_source_a) != d_a:
            raise ValueError("Precondition failed: domain A dimensions mismatch")

        if len(fake_target_b) != d_b or len(cycle_recon_b) != d_b:
            raise ValueError("Precondition failed: domain B dimensions mismatch")

        # Cycle loss: ||real_a - cycle_a||_1 + ||real_b - cycle_b||_1
        loss_cycle_a = sum(abs(real_source_a[i] - cycle_recon_a[i]) for i in range(d_a)) / float(d_a)
        loss_cycle_b = sum(abs(real_target_b[j] - cycle_recon_b[j]) for j in range(d_b)) / float(d_b)
        cycle_consistency_loss = (loss_cycle_a + loss_cycle_b) * 0.5

        # Paired L1 loss: ||real_b - fake_b||_1
        l1_paired_loss = sum(abs(real_target_b[j] - fake_target_b[j]) for j in range(d_b)) / float(d_b)

        total_objective = lambda_cycle * cycle_consistency_loss
        if is_paired:
            total_objective += lambda_l1 * l1_paired_loss

        return {
            "cycle_consistency_loss": cycle_consistency_loss,
            "l1_paired_loss": l1_paired_loss,
            "total_generator_objective": total_objective,
        }
