from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoVariationalAutoencoder:
    """
    ---
    contract:
      algo_id: ALGO-NN-152
      name: NnAlgoVariationalAutoencoder
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - vae
        - latent_variable
        - reparameterization_trick
      inputs:
        type: object
        required:
          - input_features
          - mean_weights
          - mean_bias
          - logvar_weights
          - logvar_bias
          - noise_epsilon
          - decoder_weights
          - decoder_bias
        properties:
          input_features:
            type: array
            items:
              type: number
            description: Input observation vector x of dimension d_x.
          mean_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoder mean projection matrix W_mu of shape (d_z, d_x).
          mean_bias:
            type: array
            items:
              type: number
            description: Encoder mean bias vector b_mu of dimension d_z.
          logvar_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Encoder log-variance projection matrix W_logvar of shape (d_z, d_x).
          logvar_bias:
            type: array
            items:
              type: number
            description: Encoder log-variance bias vector b_logvar of dimension d_z.
          noise_epsilon:
            type: array
            items:
              type: number
            description: Standard Gaussian sample epsilon of dimension d_z.
          decoder_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Decoder projection matrix W_d of shape (d_x, d_z).
          decoder_bias:
            type: array
            items:
              type: number
            description: Decoder bias vector b_d of dimension d_x.
          beta:
            type: number
            default: 1.0
            description: Beta-VAE capacity regularization scaling factor.
      outputs:
        type: object
        required:
          - reconstruction
          - latent_mean
          - latent_logvar
          - latent_z
          - reconstruction_loss
          - kl_divergence
          - total_loss
        properties:
          reconstruction:
            type: array
            items:
              type: number
            description: Decoded reconstruction x_hat of dimension d_x.
          latent_mean:
            type: array
            items:
              type: number
            description: Posterior mean vector mu of dimension d_z.
          latent_logvar:
            type: array
            items:
              type: number
            description: Posterior log-variance vector log(sigma^2) of dimension d_z.
          latent_z:
            type: array
            items:
              type: number
            description: Sampled continuous latent vector z of dimension d_z.
          reconstruction_loss:
            type: number
            description: Mean squared reconstruction error.
          kl_divergence:
            type: number
            description: Analytical Kullback-Leibler divergence from standard Gaussian prior.
          total_loss:
            type: number
            description: Total ELBO loss (reconstruction + beta * KL).
      parameters: {}
      input_assumptions:
        - input_features has positive length d_x
        - mean_weights and logvar_weights have shape (d_z, d_x)
        - noise_epsilon has dimension d_z
        - decoder_weights has shape (d_x, d_z)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact analytical computation within float precision"
      uses_model: false
      complexity:
        variables:
          d_x: observation dimension
          d_z: latent bottleneck dimension
        time_worst: O(d_x * d_z)
        time_typical: O(d_x * d_z)
        space: O(d_x + d_z)
      preconditions:
        - len(input.input_features) > 0
        - len(input.mean_weights) > 0 and len(input.mean_weights[0]) == len(input.input_features)
        - len(input.mean_bias) == len(input.mean_weights)
        - len(input.logvar_weights) == len(input.mean_weights) and len(input.logvar_weights[0]) == len(input.input_features)
        - len(input.logvar_bias) == len(input.mean_weights)
        - len(input.noise_epsilon) == len(input.mean_weights)
        - len(input.decoder_weights) == len(input.input_features) and len(input.decoder_weights[0]) == len(input.mean_weights)
        - len(input.decoder_bias) == len(input.input_features)
      postconditions:
        - len(output.reconstruction) == len(input.input_features)
        - len(output.latent_z) == len(input.noise_epsilon)
        - output.kl_divergence >= 0.0
      certificate: "Total loss equals reconstruction_loss + beta * kl_divergence"
      compatible_adapters:
        - ADAPTER-GENERATIVE-SAMPLER
        - ADAPTER-LATENT-COMPRESSOR
      related_algos:
        - ALGO-NN-151
        - ALGO-NN-153
      references:
        - "https://arxiv.org/abs/1312.6114"
    ---
    """

    @staticmethod
    def forward(
        input_features: List[float],
        mean_weights: List[List[float]],
        mean_bias: List[float],
        logvar_weights: List[List[float]],
        logvar_bias: List[float],
        noise_epsilon: List[float],
        decoder_weights: List[List[float]],
        decoder_bias: List[float],
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        d_x = len(input_features)
        d_z = len(noise_epsilon)
        if d_x == 0 or d_z == 0:
            raise ValueError("Precondition failed: input_features and noise_epsilon must be non-empty")

        if len(mean_weights) != d_z or any(len(r) != d_x for r in mean_weights):
            raise ValueError("Precondition failed: mean_weights dimensions invalid")
        if len(logvar_weights) != d_z or any(len(r) != d_x for r in logvar_weights):
            raise ValueError("Precondition failed: logvar_weights dimensions invalid")
        if len(mean_bias) != d_z or len(logvar_bias) != d_z:
            raise ValueError("Precondition failed: encoder bias dimensions invalid")
        if len(decoder_weights) != d_x or any(len(r) != d_z for r in decoder_weights):
            raise ValueError("Precondition failed: decoder_weights dimensions invalid")
        if len(decoder_bias) != d_x:
            raise ValueError("Precondition failed: decoder_bias dimensions invalid")

        # Compute mu and logvar
        mu = []
        logvar = []
        for i in range(d_z):
            m_val = mean_bias[i]
            lv_val = logvar_bias[i]
            for j in range(d_x):
                m_val += mean_weights[i][j] * input_features[j]
                lv_val += logvar_weights[i][j] * input_features[j]
            mu.append(m_val)
            logvar.append(lv_val)

        # Reparameterization trick: z = mu + sigma * epsilon = mu + exp(0.5 * logvar) * epsilon
        z = []
        for i in range(d_z):
            std = math.exp(0.5 * logvar[i])
            z.append(mu[i] + std * noise_epsilon[i])

        # Decode: x_hat = W_d * z + b_d
        reconstruction = []
        for i in range(d_x):
            val = decoder_bias[i]
            for j in range(d_z):
                val += decoder_weights[i][j] * z[j]
            reconstruction.append(val)

        # Reconstruction loss (MSE)
        recon_loss = sum((input_features[i] - reconstruction[i]) ** 2 for i in range(d_x)) / float(d_x)

        # Analytical KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = 0.0
        for i in range(d_z):
            kl_div += -0.5 * (1.0 + logvar[i] - (mu[i] ** 2) - math.exp(logvar[i]))
        if kl_div < 0.0:
            kl_div = 0.0

        total_loss = recon_loss + beta * kl_div

        return {
            "reconstruction": reconstruction,
            "latent_mean": mu,
            "latent_logvar": logvar,
            "latent_z": z,
            "reconstruction_loss": recon_loss,
            "kl_divergence": kl_div,
            "total_loss": total_loss,
        }
