from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoFederatedLearningFedavg:
    """
    ---
    contract:
      algo_id: ALGO-NN-246
      name: NnAlgoFederatedLearningFedavg
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - federated_learning
        - fedavg
        - distributed_optimization
        - privacy_preserving
        - client_weighting
      inputs:
        type: object
        required:
          - client_model_updates
          - client_sample_counts
          - global_weights
        properties:
          client_model_updates:
            type: array
            items:
              type: array
              items:
                type: number
            description: Locally fine-tuned parameter deltas delta_k from K participating clients.
          client_sample_counts:
            type: array
            items:
              type: integer
            description: Number of local training dataset samples n_k on each client.
          global_weights:
            type: array
            items:
              type: number
            description: Central server baseline model parameters theta_t of dimension P.
          server_learning_rate:
            type: number
            default: 1.0
            description: Server-side optimization step size eta_g.
      outputs:
        type: object
        required:
          - updated_global_weights
          - aggregated_delta
          - client_weight_proportions
          - total_samples_represented
          - client_count
        properties:
          updated_global_weights:
            type: array
            items:
              type: number
            description: Consolidated global server weights theta_{t+1} = theta_t + eta_g * Delta.
          aggregated_delta:
            type: array
            items:
              type: number
            description: Weighted average client parameter update Delta = sum_k (n_k / N) * delta_k.
          client_weight_proportions:
            type: array
            items:
              type: number
            description: Client contribution weights w_k = n_k / N.
          total_samples_represented:
            type: integer
            description: Total cumulative training data instances represented N = sum_k n_k.
          client_count:
            type: integer
            description: Number of participating federated client nodes K.
      complexity:
        time: O(K * P)
        space: O(P)
      preconditions:
        - len(input.client_model_updates) > 0
        - len(input.client_model_updates) == len(input.client_sample_counts)
        - all(len(u) == len(input.global_weights) for u in input.client_model_updates)
        - all(n > 0 for n in input.client_sample_counts)
        - input.server_learning_rate > 0.0
      postconditions:
        - len(output.updated_global_weights) == len(input.global_weights)
        - len(output.aggregated_delta) == len(input.global_weights)
        - abs(sum(output.client_weight_proportions) - 1.0) < 1e-5
        - output.total_samples_represented == sum(input.client_sample_counts)
      certificate: "Aggregated update equals exact sample-weighted convex combination of client updates"
      compatible_adapters:
        - ADAPTER-FEDAVG-SERVER-AGGREGATOR
        - ADAPTER-FEDERATED-ROBUST-SYNCHRONIZER
      related_algos:
        - ALGO-NN-233
        - ALGO-NN-247
      references:
        - "https://arxiv.org/abs/1602.05629"
        - "https://arxiv.org/abs/1812.01097"
    ---
    """

    @classmethod
    def forward(
        cls,
        client_model_updates: List[List[float]],
        client_sample_counts: List[int],
        global_weights: List[float],
        server_learning_rate: float = 1.0,
    ) -> Dict[str, Any]:
        num_clients = len(client_model_updates)
        if num_clients == 0:
            raise ValueError("Precondition failed: client_model_updates cannot be empty")
        if len(client_sample_counts) != num_clients:
            raise ValueError("Precondition failed: client updates and sample counts lengths must match")
        p = len(global_weights)
        if p == 0 or any(len(u) != p for u in client_model_updates):
            raise ValueError("Precondition failed: parameter dimension mismatch")
        if any(n <= 0 for n in client_sample_counts):
            raise ValueError("Precondition failed: sample counts must be positive")
        if server_learning_rate <= 0.0:
            raise ValueError("Precondition failed: server_learning_rate must be positive")

        # 1. Total samples N
        total_samples = sum(client_sample_counts)
        inv_total = 1.0 / float(total_samples)

        # 2. Client proportions w_k = n_k / N
        client_proportions = [float(n) * inv_total for n in client_sample_counts]

        # 3. Weighted aggregation: Delta = sum_k w_k * delta_k
        agg_delta = [0.0] * p
        for k in range(num_clients):
            w_k = client_proportions[k]
            u_k = client_model_updates[k]
            for j in range(p):
                agg_delta[j] += w_k * u_k[j]

        # 4. Server update: theta_{t+1} = theta_t + eta_g * Delta
        updated_weights: List[float] = []
        for j in range(p):
            updated_weights.append(global_weights[j] + server_learning_rate * agg_delta[j])

        return {
            "updated_global_weights": updated_weights,
            "aggregated_delta": agg_delta,
            "client_weight_proportions": client_proportions,
            "total_samples_represented": total_samples,
            "client_count": num_clients,
        }
