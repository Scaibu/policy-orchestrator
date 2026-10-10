from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoVisionTransformer:
    """
    ---
    contract:
      algo_id: ALGO-NN-128
      name: NnAlgoVisionTransformer
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - vit
        - vision_transformer
        - patch_projection
      inputs:
        type: object
        required:
          - image
          - patch_size
          - embed_dim
        properties:
          image:
            type: array
            items:
              type: array
              items:
                type: number
            description: 2D image matrix of shape (H, W).
          patch_size:
            type: integer
            minimum: 1
            description: Side length of square patch P.
          embed_dim:
            type: integer
            minimum: 1
            description: Embedding projection dimension d_model.
      outputs:
        type: object
        required:
          - patch_tokens
          - num_patches
          - token_dim
        properties:
          patch_tokens:
            type: array
            items:
              type: array
              items:
                type: number
            description: Sequence of projected patch tokens of shape (N_patches + 1, embed_dim) including CLS.
          num_patches:
            type: integer
            description: Total count of patches (H/P * W/P).
          token_dim:
            type: integer
            description: Projection embedding dimension.
      parameters: {}
      input_assumptions:
        - H and W are divisible by patch_size
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact patch linear projection"
      uses_model: false
      complexity:
        variables:
          H: height
          W: width
          P: patch_size
          d: embed_dim
        time_worst: O((H * W) + (H * W / P^2) * d)
        time_typical: O((H * W) + (H * W / P^2) * d)
        space: O((H * W / P^2) * d)
      preconditions:
        - len(input.image) > 0 and len(input.image) % input.patch_size == 0
        - len(input.image[0]) % input.patch_size == 0
      postconditions:
        - len(output.patch_tokens) == output.num_patches + 1
      certificate: "num_patches == (H // patch_size) * (W // patch_size)"
      compatible_adapters:
        - ADAPTER-VIT-PATCHIFY
      related_algos:
        - ALGO-NN-107
        - ALGO-NN-112
      references:
        - "https://arxiv.org/abs/2010.11929"
    ---
    """

    @staticmethod
    def patchify_and_embed(
        image: List[List[float]],
        patch_size: int = 16,
        embed_dim: int = 64,
    ) -> Dict[str, Any]:
        if not image or patch_size < 1 or embed_dim < 1:
            raise ValueError("Precondition failed: invalid inputs")

        H = len(image)
        W = len(image[0])
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError("Precondition failed: image dims must be divisible by patch_size")

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w

        cls_token = [0.1] * embed_dim
        tokens: List[List[float]] = [cls_token]

        for ph in range(num_patches_h):
            for pw in range(num_patches_w):
                patch_pixels: List[float] = []
                for r in range(ph * patch_size, (ph + 1) * patch_size):
                    for c in range(pw * patch_size, (pw + 1) * patch_size):
                        patch_pixels.append(image[r][c])

                tok = [sum(patch_pixels[k] * 0.01 for k in range(len(patch_pixels))) + float(d) * 0.001 for d in range(embed_dim)]
                tokens.append(tok)

        return {
            "patch_tokens": tokens,
            "num_patches": total_patches,
            "token_dim": embed_dim,
        }
