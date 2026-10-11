from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoAutoregressivePixelcnn:
    """
    ---
    contract:
      algo_id: ALGO-NN-159
      name: NnAlgoAutoregressivePixelcnn
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - pixelcnn
        - autoregressive
        - causal_convolution
        - density_estimation
      inputs:
        type: object
        required:
          - pixel_sequence
          - causal_weights
          - causal_bias
          - vocab_size
        properties:
          pixel_sequence:
            type: array
            items:
              type: integer
            description: Ordered sequence of raster discrete pixel values of length N.
          causal_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Causal projection weights of shape (vocab_size, max_context_len).
          causal_bias:
            type: array
            items:
              type: number
            description: Causal projection bias of dimension vocab_size.
          vocab_size:
            type: integer
            minimum: 2
            description: Number of discrete pixel quantization bins V.
      outputs:
        type: object
        required:
          - total_nll
          - mean_nll_per_pixel
          - bits_per_dim
          - predicted_probabilities
        properties:
          total_nll:
            type: number
            description: Total negative log-likelihood over entire raster sequence.
          mean_nll_per_pixel:
            type: number
            description: Average negative log-likelihood per pixel.
          bits_per_dim:
            type: number
            description: Compression metric in bits per dimension (nats / ln(2)).
          predicted_probabilities:
            type: array
            items:
              type: array
              items:
                type: number
            description: Categorical probability distributions per raster step of shape (N, vocab_size).
      parameters: {}
      input_assumptions:
        - pixel_sequence is non-empty with values in [0, vocab_size - 1]
        - causal_weights has shape (vocab_size, max_context) where max_context >= len(pixel_sequence)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized softmax with max-subtraction"
      uses_model: false
      complexity:
        variables:
          N: sequence length
          V: vocabulary size
        time_worst: O(N^2 * V)
        time_typical: O(N^2 * V)
        space: O(N * V)
      preconditions:
        - len(input.pixel_sequence) > 0
        - input.vocab_size >= 2
        - all(0 <= p < input.vocab_size for p in input.pixel_sequence)
        - len(input.causal_weights) == input.vocab_size
        - len(input.causal_bias) == input.vocab_size
        - len(input.causal_weights[0]) >= len(input.pixel_sequence)
      postconditions:
        - output.total_nll >= 0.0
        - output.bits_per_dim >= 0.0
        - len(output.predicted_probabilities) == len(input.pixel_sequence)
      certificate: "Bits per dimension equals mean_nll_per_pixel divided by natural log of 2"
      compatible_adapters:
        - ADAPTER-PIXEL-GENERATOR
        - ADAPTER-IMAGE-DENSITY-ESTIMATOR
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-158
      references:
        - "https://arxiv.org/abs/1606.05328"
    ---
    """

    @staticmethod
    def forward(
        pixel_sequence: List[int],
        causal_weights: List[List[float]],
        causal_bias: List[float],
        vocab_size: int,
    ) -> Dict[str, Any]:
        N = len(pixel_sequence)
        if N == 0:
            raise ValueError("Precondition failed: pixel_sequence cannot be empty")
        if vocab_size < 2:
            raise ValueError("Precondition failed: vocab_size must be at least 2")
        if any(p < 0 or p >= vocab_size for p in pixel_sequence):
            raise ValueError("Precondition failed: pixel values out of range")
        if len(causal_weights) != vocab_size or len(causal_bias) != vocab_size:
            raise ValueError("Precondition failed: weights and bias must match vocab_size")
        if len(causal_weights[0]) < N:
            raise ValueError("Precondition failed: causal_weights context dimension insufficient")

        total_nll = 0.0
        predicted_probs: List[List[float]] = []
        eps = 1e-12

        for i in range(N):
            # Compute logits strictly from causal history: indices 0 to i-1
            # For position 0, causal history is empty (prior bias only)
            logits: List[float] = []
            for v in range(vocab_size):
                logit = causal_bias[v]
                for past_idx in range(i):
                    logit += causal_weights[v][past_idx] * float(pixel_sequence[past_idx])
                logits.append(logit)

            # Numerically stable softmax
            max_logit = max(logits)
            exp_sum = sum(math.exp(l - max_logit) for l in logits)
            probs = [math.exp(l - max_logit) / exp_sum for l in logits]
            predicted_probs.append(probs)

            # Target likelihood for pixel i
            target_pixel = pixel_sequence[i]
            target_prob = max(eps, probs[target_pixel])
            total_nll += -math.log(target_prob)

        mean_nll = total_nll / float(N)
        bpd = mean_nll / math.log(2.0)

        return {
            "total_nll": total_nll,
            "mean_nll_per_pixel": mean_nll,
            "bits_per_dim": bpd,
            "predicted_probabilities": predicted_probs,
        }
