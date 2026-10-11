from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMaskedAutoencodersVision:
    """
    ---
    contract:
      algo_id: ALGO-NN-174
      name: NnAlgoMaskedAutoencodersVision
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_supervised_learning
        - vision_transformer
        - masked_autoencoder
        - mae
        - asymmetric_encoder_decoder
      inputs:
        type: object
        required:
          - original_patches
          - reconstructed_patches
          - mask_indicators
        properties:
          original_patches:
            type: array
            items:
              type: array
              items:
                type: number
            description: Ground truth flattened image patches of shape (N_patches, d_patch).
          reconstructed_patches:
            type: array
            items:
              type: array
              items:
                type: number
            description: Decoder predicted patch reconstructions of shape (N_patches, d_patch).
          mask_indicators:
            type: array
            items:
              type: integer
            description: Binary mask indicators of length N_patches (1 = masked/held-out, 0 = visible).
          normalize_pixels:
            type: boolean
            default: true
            description: Whether to compute loss on per-patch mean/variance normalized pixels.
      outputs:
        type: object
        required:
          - masked_patch_loss
          - overall_reconstruction_loss
          - num_masked_patches
          - num_visible_patches
          - masking_ratio
        properties:
          masked_patch_loss:
            type: number
            description: Mean squared error computed strictly over masked patch positions.
          overall_reconstruction_loss:
            type: number
            description: Mean squared error computed across all patches.
          num_masked_patches:
            type: integer
            description: Total count of masked patches.
          num_visible_patches:
            type: integer
            description: Total count of visible encoder patches.
          masking_ratio:
            type: number
            description: Empirical masking fraction |M| / N.
      parameters: {}
      input_assumptions:
        - original_patches and reconstructed_patches have matching non-empty dimensions (N, d_patch)
        - mask_indicators has length N with binary values {0, 1}
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact per-patch normalized MSE"
      uses_model: false
      complexity:
        variables:
          N: number of patches
          D: patch pixel dimension
        time_worst: O(N * D)
        time_typical: O(N * D)
        space: O(N * D)
      preconditions:
        - len(input.original_patches) > 0 and len(input.original_patches[0]) > 0
        - len(input.original_patches) == len(input.reconstructed_patches) == len(input.mask_indicators)
        - len(input.original_patches[0]) == len(input.reconstructed_patches[0])
      postconditions:
        - output.masked_patch_loss >= 0.0
        - output.overall_reconstruction_loss >= 0.0
        - output.num_masked_patches + output.num_visible_patches == len(input.original_patches)
      certificate: "Masked patch loss isolates reconstruction strictly to mask_indicators == 1"
      compatible_adapters:
        - ADAPTER-MAE-PRETRAINER
        - ADAPTER-VIT-MASKED-MODEL
      related_algos:
        - ALGO-NN-104
        - ALGO-NN-151
        - ALGO-NN-176
      references:
        - "https://arxiv.org/abs/2111.06377"
    ---
    """

    @classmethod
    def forward(
        cls,
        original_patches: List[List[float]],
        reconstructed_patches: List[List[float]],
        mask_indicators: List[int],
        normalize_pixels: bool = True,
    ) -> Dict[str, Any]:
        N = len(original_patches)
        if N == 0 or len(original_patches[0]) == 0:
            raise ValueError("Precondition failed: original_patches cannot be empty")
        d_patch = len(original_patches[0])

        if len(reconstructed_patches) != N or any(len(r) != d_patch for r in original_patches) or any(len(r) != d_patch for r in reconstructed_patches):
            raise ValueError("Precondition failed: uniform patch matrix dimensions required")
        if len(mask_indicators) != N or any(m not in (0, 1) for m in mask_indicators):
            raise ValueError("Precondition failed: mask_indicators must be binary of length N")

        eps = 1e-6
        masked_sq_error_sum = 0.0
        total_sq_error_sum = 0.0
        n_masked = 0
        n_visible = 0

        for i in range(N):
            orig_row = original_patches[i]
            recon_row = reconstructed_patches[i]
            is_masked = (mask_indicators[i] == 1)

            if is_masked:
                n_masked += 1
            else:
                n_visible += 1

            if normalize_pixels:
                mean_p = sum(orig_row) / float(d_patch)
                var_p = sum((x - mean_p) ** 2 for x in orig_row) / float(d_patch)
                std_p = math.sqrt(var_p + eps)
                target_row = [(x - mean_p) / std_p for x in orig_row]
            else:
                target_row = orig_row

            patch_err = 0.0
            for j in range(d_patch):
                diff = recon_row[j] - target_row[j]
                patch_err += diff * diff

            total_sq_error_sum += patch_err
            if is_masked:
                masked_sq_error_sum += patch_err

        masked_loss = (masked_sq_error_sum / float(n_masked * d_patch)) if n_masked > 0 else 0.0
        total_loss = total_sq_error_sum / float(N * d_patch)
        ratio = float(n_masked) / float(N)

        return {
            "masked_patch_loss": masked_loss,
            "overall_reconstruction_loss": total_loss,
            "num_masked_patches": n_masked,
            "num_visible_patches": n_visible,
            "masking_ratio": ratio,
        }
