from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDenoisingAutoencoder:
    """
    ---
    contract:
      algo_id: ALGO-NN-151
      name: NnAlgoDenoisingAutoencoder
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - autoencoder
        - denoising
        - representation_learning
      inputs:
        type: object
        required:
          - clean_input
          - corrupted_input
          - encoder_weights
          - encoder_bias
          - decoder_weights
          - decoder_bias
        properties:
          clean_input:
            type: array
            items:
              type: number
            description: Clean target feature vector x of dimension d_x.
          corrupted_input:
            type: array
            items:
              type: number
            description: Corrupted or noisy feature vector x_tilde of dimension d_x.
          encoder_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoder weight matrix W_e of shape (d_h, d_x).
          encoder_bias:
            type: array
            items:
              type: number
            description: Encoder bias vector b_e of dimension d_h.
          decoder_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Decoder weight matrix W_d of shape (d_x, d_h).
          decoder_bias:
            type: array
            items:
              type: number
            description: Decoder bias vector b_d of dimension d_x.
      outputs:
        type: object
        required:
          - reconstruction
          - reconstruction_loss
          - latent_representation
          - anomaly_score
        properties:
          reconstruction:
            type: array
            items:
              type: number
            description: Denoised reconstructed feature vector x_hat.
          reconstruction_loss:
            type: number
            description: Mean squared reconstruction error against clean input.
          latent_representation:
            type: array
            items:
              type: number
            description: Latent bottleneck representation vector h.
          anomaly_score:
            type: number
            description: Normalized residual reconstruction energy for anomaly flagging.
      parameters: {}
      input_assumptions:
        - clean_input and corrupted_input have identical positive dimension d_x
        - encoder_weights has shape (d_h, d_x) and decoder_weights has shape (d_x, d_h)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact within IEEE-754 precision"
      uses_model: false
      complexity:
        variables:
          d_x: input feature dimension
          d_h: hidden latent dimension
        time_worst: O(d_x * d_h)
        time_typical: O(d_x * d_h)
        space: O(d_x + d_h)
      preconditions:
        - len(input.clean_input) > 0
        - len(input.clean_input) == len(input.corrupted_input)
        - len(input.encoder_weights) > 0 and len(input.encoder_weights[0]) == len(input.clean_input)
        - len(input.encoder_bias) == len(input.encoder_weights)
        - len(input.decoder_weights) == len(input.clean_input) and len(input.decoder_weights[0]) == len(input.encoder_bias)
        - len(input.decoder_bias) == len(input.clean_input)
      postconditions:
        - len(output.reconstruction) == len(input.clean_input)
        - len(output.latent_representation) == len(input.encoder_bias)
        - output.reconstruction_loss >= 0.0
      certificate: "Reconstruction loss equals mean squared error between clean input and reconstructed output"
      compatible_adapters:
        - ADAPTER-ANOMALY-DETECTOR
        - ADAPTER-FEATURE-EXTRACTOR
      related_algos:
        - ALGO-NN-152
        - ALGO-NN-104
      references:
        - "https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf"
    ---
    """

    @staticmethod
    def forward(
        clean_input: List[float],
        corrupted_input: List[float],
        encoder_weights: List[List[float]],
        encoder_bias: List[float],
        decoder_weights: List[List[float]],
        decoder_bias: List[float],
    ) -> Dict[str, Any]:
        d_x = len(clean_input)
        if d_x == 0 or len(corrupted_input) != d_x:
            raise ValueError("Precondition failed: clean and corrupted inputs must have matching positive length")

        d_h = len(encoder_bias)
        if len(encoder_weights) != d_h or any(len(row) != d_x for row in encoder_weights):
            raise ValueError("Precondition failed: encoder_weights dimension mismatch")

        if len(decoder_weights) != d_x or any(len(row) != d_h for row in decoder_weights):
            raise ValueError("Precondition failed: decoder_weights dimension mismatch")

        if len(decoder_bias) != d_x:
            raise ValueError("Precondition failed: decoder_bias dimension mismatch")

        # Encode: h = ReLU(W_e * x_tilde + b_e)
        latent = []
        for i in range(d_h):
            activation = encoder_bias[i]
            for j in range(d_x):
                activation += encoder_weights[i][j] * corrupted_input[j]
            latent.append(activation if activation > 0.0 else 0.0)

        # Decode: x_hat = W_d * h + b_d
        reconstruction = []
        for i in range(d_x):
            val = decoder_bias[i]
            for j in range(d_h):
                val += decoder_weights[i][j] * latent[j]
            reconstruction.append(val)

        # Compute Mean Squared Reconstruction Error
        mse_sum = 0.0
        for i in range(d_x):
            diff = clean_input[i] - reconstruction[i]
            mse_sum += diff * diff

        loss = mse_sum / float(d_x)

        return {
            "reconstruction": reconstruction,
            "reconstruction_loss": loss,
            "latent_representation": latent,
            "anomaly_score": loss,
        }
