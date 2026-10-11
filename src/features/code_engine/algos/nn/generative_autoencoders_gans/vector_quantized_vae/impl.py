from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoVectorQuantizedVae:
    """
    ---
    contract:
      algo_id: ALGO-NN-153
      name: NnAlgoVectorQuantizedVae
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - vq_vae
        - discrete_latent
        - vector_quantization
      inputs:
        type: object
        required:
          - encoder_outputs
          - codebook
        properties:
          encoder_outputs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Continuous latent representations z_e of shape (T, D).
          codebook:
            type: array
            items:
              type: array
              items:
                type: number
            description: Discrete learned codebook matrix E of shape (K, D).
          beta:
            type: number
            default: 0.25
            description: Commitment cost hyperparameter.
      outputs:
        type: object
        required:
          - quantized_vectors
          - codebook_indices
          - commitment_loss
          - codebook_loss
          - quantization_loss
          - perplexity
        properties:
          quantized_vectors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Nearest codebook replacement vectors z_q of shape (T, D).
          codebook_indices:
            type: array
            items:
              type: integer
            description: Selected discrete codebook index sequence of length T.
          commitment_loss:
            type: number
            description: Encoder commitment penalty to prevent latent drift.
          codebook_loss:
            type: number
            description: Dictionary learning loss pulling codes toward encoder states.
          quantization_loss:
            type: number
            description: Combined VQ loss (codebook_loss + beta * commitment_loss).
          perplexity:
            type: number
            description: Codebook usage entropy perplexity metric.
      parameters: {}
      input_assumptions:
        - encoder_outputs is non-empty with uniform embedding dimension D
        - codebook is non-empty with uniform embedding dimension D
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact Euclidean nearest-neighbor lookup"
      uses_model: false
      complexity:
        variables:
          T: sequence length
          K: codebook size
          D: embedding dimension
        time_worst: O(T * K * D)
        time_typical: O(T * K * D)
        space: O(T * D)
      preconditions:
        - len(input.encoder_outputs) > 0 and len(input.encoder_outputs[0]) > 0
        - len(input.codebook) > 0 and len(input.codebook[0]) == len(input.encoder_outputs[0])
        - input.beta >= 0.0
      postconditions:
        - len(output.quantized_vectors) == len(input.encoder_outputs)
        - len(output.codebook_indices) == len(input.encoder_outputs)
        - output.quantization_loss >= 0.0
        - output.perplexity >= 1.0
      certificate: "Each quantized vector strictly belongs to the input codebook rows"
      compatible_adapters:
        - ADAPTER-DISCRETE-TOKENIZER
        - ADAPTER-LATENT-QUANTIZER
      related_algos:
        - ALGO-NN-152
        - ALGO-NN-166
      references:
        - "https://arxiv.org/abs/1711.00937"
    ---
    """

    @staticmethod
    def forward(
        encoder_outputs: List[List[float]],
        codebook: List[List[float]],
        beta: float = 0.25,
    ) -> Dict[str, Any]:
        T = len(encoder_outputs)
        if T == 0 or len(encoder_outputs[0]) == 0:
            raise ValueError("Precondition failed: encoder_outputs must be non-empty")

        D = len(encoder_outputs[0])
        if any(len(row) != D for row in encoder_outputs):
            raise ValueError("Precondition failed: non-uniform encoder output dimension")

        K = len(codebook)
        if K == 0 or any(len(row) != D for row in codebook):
            raise ValueError("Precondition failed: codebook dimension mismatch")

        quantized_vectors: List[List[float]] = []
        codebook_indices: List[int] = []
        counts: Dict[int, int] = {k: 0 for k in range(K)}

        total_commitment_sq = 0.0
        total_codebook_sq = 0.0

        for t in range(T):
            z_e = encoder_outputs[t]
            best_idx = 0
            best_dist = float("inf")

            for k in range(K):
                dist = sum((z_e[d] - codebook[k][d]) ** 2 for d in range(D))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = k

            z_q = list(codebook[best_idx])
            quantized_vectors.append(z_q)
            codebook_indices.append(best_idx)
            counts[best_idx] += 1

            # Loss components
            sq_diff = sum((z_e[d] - z_q[d]) ** 2 for d in range(D))
            total_commitment_sq += sq_diff
            total_codebook_sq += sq_diff

        commitment_loss = total_commitment_sq / float(T)
        codebook_loss = total_codebook_sq / float(T)
        quantization_loss = codebook_loss + beta * commitment_loss

        # Compute codebook usage perplexity: exp(-sum(p * log(p)))
        entropy = 0.0
        for k in range(K):
            if counts[k] > 0:
                p_k = float(counts[k]) / float(T)
                entropy -= p_k * math.log(p_k)
        perplexity = math.exp(entropy)

        return {
            "quantized_vectors": quantized_vectors,
            "codebook_indices": codebook_indices,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "quantization_loss": quantization_loss,
            "perplexity": perplexity,
        }
