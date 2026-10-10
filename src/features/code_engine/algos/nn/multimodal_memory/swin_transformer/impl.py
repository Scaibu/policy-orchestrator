from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSwinTransformer:
    """
    ---
    contract:
      algo_id: ALGO-NN-129
      name: NnAlgoSwinTransformer
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - swin_transformer
        - shifted_window
        - hierarchical_vision
      inputs:
        type: object
        required:
          - feature_map
          - window_size
          - shift_size
        properties:
          feature_map:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: 3D feature grid of shape (H, W, C).
          window_size:
            type: integer
            minimum: 1
            description: Window side length M.
          shift_size:
            type: integer
            minimum: 0
            description: Window cyclic shift offset.
      outputs:
        type: object
        required:
          - partitioned_windows
          - num_windows
          - window_tokens
        properties:
          partitioned_windows:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: List of window blocks of shape (num_windows, window_size * window_size, C).
          num_windows:
            type: integer
            description: Total window count (H/M * W/M).
          window_tokens:
            type: integer
            description: Tokens per window M^2.
      parameters: {}
      input_assumptions:
        - H and W divisible by window_size
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact window partitioning"
      uses_model: false
      complexity:
        variables:
          H: height
          W: width
          C: channels
          M: window_size
        time_worst: O(H * W * C)
        time_typical: O(H * W * C)
        space: O(H * W * C)
      preconditions:
        - len(input.feature_map) > 0
        - len(input.feature_map) % input.window_size == 0
      postconditions:
        - len(output.partitioned_windows) == output.num_windows
      certificate: "num_windows == (H // window_size) * (W // window_size)"
      compatible_adapters:
        - ADAPTER-SWIN-WINDOW
      related_algos:
        - ALGO-NN-128
      references:
        - "https://arxiv.org/abs/2103.14030"
    ---
    """

    @staticmethod
    def partition_windows(
        feature_map: List[List[List[float]]],
        window_size: int = 7,
        shift_size: int = 0,
    ) -> Dict[str, Any]:
        if not feature_map or window_size < 1 or shift_size < 0:
            raise ValueError("Precondition failed: invalid parameters")

        H = len(feature_map)
        W = len(feature_map[0])
        C = len(feature_map[0][0])
        if H % window_size != 0 or W % window_size != 0:
            raise ValueError("Precondition failed: grid dims must be divisible by window_size")

        shifted_map: List[List[List[float]]] = [[feature_map[(r + shift_size) % H][(c + shift_size) % W][:] for c in range(W)] for r in range(H)]

        num_win_h = H // window_size
        num_win_w = W // window_size
        windows: List[List[List[float]]] = []

        for wh in range(num_win_h):
            for ww in range(num_win_w):
                win_tokens: List[List[float]] = []
                for r in range(wh * window_size, (wh + 1) * window_size):
                    for c in range(ww * window_size, (ww + 1) * window_size):
                        win_tokens.append(shifted_map[r][c][:])
                windows.append(win_tokens)

        return {
            "partitioned_windows": windows,
            "num_windows": len(windows),
            "window_tokens": window_size * window_size,
        }
