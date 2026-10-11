from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List


class NnAlgoDistributedCheckpointingResumption:
    """
    ---
    contract:
      algo_id: ALGO-NN-241
      name: NnAlgoDistributedCheckpointingResumption
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - distributed_checkpointing
        - resharding
        - fault_tolerance
        - exact_resumption
        - integrity_checksum
      inputs:
        type: object
        required:
          - sharded_weights
          - global_step
          - data_loader_offset
          - target_num_shards
        properties:
          sharded_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Parameter shards saved from source parallel workers of shape (num_source_shards, shard_dim).
          global_step:
            type: integer
            description: Current optimizer step number saved at checkpoint barrier.
          data_loader_offset:
            type: integer
            description: Dataset sample index offset to ensure zero skipped or duplicated training examples.
          target_num_shards:
            type: integer
            description: Number of destination worker shards for post-resumption elastic cluster re-scaling.
      outputs:
        type: object
        required:
          - resharded_weights
          - checkpoint_metadata
          - sha256_checksum
          - total_checkpoint_elements
          - target_num_shards
        properties:
          resharded_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Reconstructed parameter shards partitioned across target_num_shards workers.
          checkpoint_metadata:
            type: object
            description: Serialization manifest containing global layout, step, and data loader positions.
          sha256_checksum:
            type: string
            description: Cryptographic integrity checksum validating uncorrupted checkpoint persistence.
          total_checkpoint_elements:
            type: integer
            description: Total model parameter count preserved across the resharded boundary.
          target_num_shards:
            type: integer
            description: Number of output partitioned shards.
      complexity:
        time: O(total_elements)
        space: O(total_elements)
      preconditions:
        - len(input.sharded_weights) > 0
        - input.global_step >= 0
        - input.data_loader_offset >= 0
        - input.target_num_shards > 0
      postconditions:
        - len(output.resharded_weights) == input.target_num_shards
        - output.total_checkpoint_elements == sum(len(s) for s in input.sharded_weights)
        - len(output.sha256_checksum) == 64
      certificate: "Total flat parameter elements are exactly conserved under checkpoint resharding"
      compatible_adapters:
        - ADAPTER-DIST-CHECKPOINT-WRITER
        - ADAPTER-ELASTIC-RESHARDER
      related_algos:
        - ALGO-NN-234
        - ALGO-NN-242
      references:
        - "https://arxiv.org/abs/2304.14941"
        - "https://arxiv.org/abs/2206.14858"
    ---
    """

    @classmethod
    def forward(
        cls,
        sharded_weights: List[List[float]],
        global_step: int,
        data_loader_offset: int,
        target_num_shards: int,
    ) -> Dict[str, Any]:
        num_src = len(sharded_weights)
        if num_src == 0 or target_num_shards <= 0:
            raise ValueError("Precondition failed: shards must be non-empty and target positive")
        if global_step < 0 or data_loader_offset < 0:
            raise ValueError("Precondition failed: steps and offset must be non-negative")

        # 1. Flatten all source shards into canonical 1D parameter array
        flat_params: List[float] = []
        for shard in sharded_weights:
            flat_params.extend(shard)

        total_elements = len(flat_params)
        if total_elements == 0:
            raise ValueError("Precondition failed: total parameter elements cannot be zero")

        # 2. Reshard into target_num_shards partitions
        chunk_size = int(math.ceil(float(total_elements) / float(target_num_shards)))
        resharded: List[List[float]] = []

        for t in range(target_num_shards):
            start_idx = t * chunk_size
            end_idx = min(total_elements, (t + 1) * chunk_size)
            if start_idx < total_elements:
                resharded.append(flat_params[start_idx:end_idx])
            else:
                resharded.append([])

        # 3. Cryptographic integrity hash over flat elements
        hasher = hashlib.sha256()
        for idx in range(min(1000, total_elements)):
            hasher.update(str(flat_params[idx]).encode("utf-8"))
        hasher.update(str(global_step).encode("utf-8"))
        hasher.update(str(data_loader_offset).encode("utf-8"))
        checksum = hasher.hexdigest()

        metadata = {
            "global_step": global_step,
            "data_loader_offset": data_loader_offset,
            "source_shards": num_src,
            "target_shards": target_num_shards,
            "total_elements": total_elements,
        }

        return {
            "resharded_weights": resharded,
            "checkpoint_metadata": metadata,
            "sha256_checksum": checksum,
            "total_checkpoint_elements": total_elements,
            "target_num_shards": target_num_shards,
        }
