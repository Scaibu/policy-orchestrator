from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoZeroFsdpShardedTraining:
    """
    ---
    contract:
      algo_id: ALGO-NN-234
      name: NnAlgoZeroFsdpShardedTraining
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - zero
        - fsdp
        - memory_sharding
        - optimizer_partitioning
        - parameter_sharding
      inputs:
        type: object
        required:
          - total_parameters
          - num_devices
        properties:
          total_parameters:
            type: integer
            description: Total model parameter count N_params.
          num_devices:
            type: integer
            description: Number of distributed GPU devices N_dev.
          zero_stage:
            type: integer
            default: 3
            enum: [1, 2, 3]
            description: ZeRO sharding tier: 1=ZeRO-Optimizer, 2=ZeRO-Grad, 3=ZeRO-Param (FSDP).
          bytes_per_parameter:
            type: integer
            default: 2
            description: Precision bytes per parameter (2 for FP16/BF16).
          optimizer_type:
            type: string
            default: adam
            enum: [adam, sgd]
            description: Optimizer state layout: Adam (12 bytes/param) vs SGD with momentum (4 bytes/param).
      outputs:
        type: object
        required:
          - memory_per_device_mb
          - memory_saving_factor
          - sharded_parameter_count_per_device
          - communication_volume_mb
          - zero_stage
        properties:
          memory_per_device_mb:
            type: number
            description: Static model state memory consumption per device in megabytes.
          memory_saving_factor:
            type: number
            description: Relative memory reduction compared to standard Data Parallelism.
          sharded_parameter_count_per_device:
            type: integer
            description: Number of parameters stored on each device's shard.
          communication_volume_mb:
            type: number
            description: Total forward and backward collective communication volume per step in MB.
          zero_stage:
            type: integer
            description: Active ZeRO/FSDP tier.
      complexity:
        time: O(1)
        space: O(1)
      preconditions:
        - input.total_parameters > 0
        - input.num_devices > 0
        - input.zero_stage in [1, 2, 3]
        - input.bytes_per_parameter > 0
      postconditions:
        - output.memory_per_device_mb > 0.0
        - output.memory_saving_factor >= 1.0
        - output.sharded_parameter_count_per_device > 0
      certificate: "Memory per device equals sum of partitioned model states according to ZeRO stage"
      compatible_adapters:
        - ADAPTER-ZERO-STAGE-CONTROLLER
        - ADAPTER-FSDP-WRAPER
      related_algos:
        - ALGO-NN-233
        - ALGO-NN-240
      references:
        - "https://arxiv.org/abs/1910.02054"
        - "https://arxiv.org/abs/2304.11277"
    ---
    """

    @classmethod
    def forward(
        cls,
        total_parameters: int,
        num_devices: int,
        zero_stage: int = 3,
        bytes_per_parameter: int = 2,
        optimizer_type: str = "adam",
    ) -> Dict[str, Any]:
        if total_parameters <= 0:
            raise ValueError("Precondition failed: total_parameters must be positive")
        if num_devices <= 0:
            raise ValueError("Precondition failed: num_devices must be positive")
        if zero_stage not in [1, 2, 3]:
            raise ValueError("Precondition failed: zero_stage must be 1, 2, or 3")
        if bytes_per_parameter <= 0:
            raise ValueError("Precondition failed: bytes_per_parameter must be positive")

        p = float(total_parameters)
        n = float(num_devices)

        # Baseline DDP Memory (Parameters in 16-bit, Gradients in 16-bit, Optimizer state in FP32)
        # For Adam: 4 bytes (FP32 master weight) + 4 bytes (momentum m) + 4 bytes (variance v) = 12 bytes/param
        opt_bytes_per_param = 12.0 if optimizer_type == "adam" else 4.0

        param_bytes = p * bytes_per_parameter
        grad_bytes = p * bytes_per_parameter
        opt_bytes = p * opt_bytes_per_param

        ddp_static_bytes = param_bytes + grad_bytes + opt_bytes

        if zero_stage == 1:
            # ZeRO-1: Shard optimizer states only
            dev_bytes = param_bytes + grad_bytes + (opt_bytes / n)
            sharded_params = total_parameters
            # All-Reduce gradients: 2 * (N-1)/N * param_bytes
            comm_bytes = 2.0 * ((n - 1.0) / n) * grad_bytes
        elif zero_stage == 2:
            # ZeRO-2: Shard optimizer states + gradients
            dev_bytes = param_bytes + (grad_bytes / n) + (opt_bytes / n)
            sharded_params = total_parameters
            # Reduce-Scatter gradients: (N-1)/N * grad_bytes
            comm_bytes = ((n - 1.0) / n) * grad_bytes
        else:
            # ZeRO-3 / FSDP: Shard optimizer states + gradients + parameters
            dev_bytes = (param_bytes / n) + (grad_bytes / n) + (opt_bytes / n)
            sharded_params = int(math.ceil(p / n))
            # Forward All-Gather + Backward All-Gather + Reduce-Scatter:
            # 2 * (N-1)/N * param_bytes (All-Gathers) + (N-1)/N * grad_bytes (Reduce-Scatter) = ~3x communication
            comm_bytes = 3.0 * ((n - 1.0) / n) * param_bytes

        mem_mb = dev_bytes / (1024.0 * 1024.0)
        saving_factor = ddp_static_bytes / dev_bytes
        comm_mb = comm_bytes / (1024.0 * 1024.0)

        return {
            "memory_per_device_mb": mem_mb,
            "memory_saving_factor": saving_factor,
            "sharded_parameter_count_per_device": sharded_params,
            "communication_volume_mb": comm_mb,
            "zero_stage": zero_stage,
        }
