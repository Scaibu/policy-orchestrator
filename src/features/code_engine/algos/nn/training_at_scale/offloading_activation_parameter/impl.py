from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoOffloadingActivationParameter:
    """
    ---
    contract:
      algo_id: ALGO-NN-240
      name: NnAlgoOffloadingActivationParameter
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - memory_offloading
        - cpu_offload
        - nvme_offload
        - zero_infinity
        - pcie_bandwidth
      inputs:
        type: object
        required:
          - gpu_memory_capacity_mb
          - total_layers
          - layer_parameter_mb
          - layer_activation_mb
        properties:
          gpu_memory_capacity_mb:
            type: number
            description: Total physical VRAM capacity on local GPU in megabytes.
          total_layers:
            type: integer
            description: Total sequential transformer layers in network architecture.
          layer_parameter_mb:
            type: number
            description: Parameter storage required per individual layer in MB.
          layer_activation_mb:
            type: number
            description: Forward activation cache required per individual layer in MB.
          current_layer_idx:
            type: integer
            default: 0
            description: Active executing layer index.
          pcie_bandwidth_gb_s:
            type: number
            default: 32.0
            description: Unidirectional PCIe Gen4/Gen5 transfer bandwidth in GB/s.
          offload_target:
            type: string
            default: cpu
            enum: [cpu, nvme]
            description: Storage tier for offloaded states (cpu RAM vs nvme SSD).
      outputs:
        type: object
        required:
          - gpu_resident_memory_mb
          - host_offloaded_memory_mb
          - layer_swap_latency_ms
          - is_offload_required
          - total_model_size_mb
        properties:
          gpu_resident_memory_mb:
            type: number
            description: Active memory resident on GPU (current layer params + activations).
          host_offloaded_memory_mb:
            type: number
            description: Memory evicted to host CPU/NVMe memory pool.
          layer_swap_latency_ms:
            type: number
            description: Estimated transfer latency to stream layer weights over PCIe.
          is_offload_required:
            type: boolean
            description: True if total monolithic model memory exceeds GPU capacity.
          total_model_size_mb:
            type: number
            description: Total unpartitioned model memory footprint across all layers.
      complexity:
        time: O(1)
        space: O(1)
      preconditions:
        - input.gpu_memory_capacity_mb > 0.0
        - input.total_layers > 0
        - input.layer_parameter_mb > 0.0
        - input.layer_activation_mb > 0.0
        - input.pcie_bandwidth_gb_s > 0.0
      postconditions:
        - output.gpu_resident_memory_mb <= input.gpu_memory_capacity_mb or not output.is_offload_required
        - output.layer_swap_latency_ms >= 0.0
        - output.total_model_size_mb > 0.0
      certificate: "Total model size equals total_layers * (layer_parameter_mb + layer_activation_mb)"
      compatible_adapters:
        - ADAPTER-ZERO-INFINITY-OFFLOADER
        - ADAPTER-ACTIVATION-CHECKPOINT-OFFLOADER
      related_algos:
        - ALGO-NN-234
        - ALGO-NN-236
      references:
        - "https://arxiv.org/abs/2104.07857"
        - "https://arxiv.org/abs/2101.06840"
    ---
    """

    @classmethod
    def forward(
        cls,
        gpu_memory_capacity_mb: float,
        total_layers: int,
        layer_parameter_mb: float,
        layer_activation_mb: float,
        current_layer_idx: int = 0,
        pcie_bandwidth_gb_s: float = 32.0,
        offload_target: str = "cpu",
    ) -> Dict[str, Any]:
        if gpu_memory_capacity_mb <= 0.0 or total_layers <= 0:
            raise ValueError("Precondition failed: GPU memory and total layers must be positive")
        if layer_parameter_mb <= 0.0 or layer_activation_mb <= 0.0:
            raise ValueError("Precondition failed: layer memory costs must be positive")
        if pcie_bandwidth_gb_s <= 0.0:
            raise ValueError("Precondition failed: PCIe bandwidth must be positive")

        total_param_mb = float(total_layers) * layer_parameter_mb
        total_act_mb = float(total_layers) * layer_activation_mb
        total_model_mb = total_param_mb + total_act_mb

        is_offload = total_model_mb > gpu_memory_capacity_mb

        if is_offload:
            # Stream only active layer parameters + activations to GPU
            gpu_resident = layer_parameter_mb + layer_activation_mb
            host_offload = total_model_mb - gpu_resident

            # Effective bandwidth in MB/s
            bandwidth_mb_s = pcie_bandwidth_gb_s * 1024.0
            if offload_target == "nvme":
                bandwidth_mb_s *= 0.25  # NVMe slower than PCIe CPU host RAM

            swap_latency_ms = (layer_parameter_mb / bandwidth_mb_s) * 1000.0
        else:
            gpu_resident = total_model_mb
            host_offload = 0.0
            swap_latency_ms = 0.0

        return {
            "gpu_resident_memory_mb": gpu_resident,
            "host_offloaded_memory_mb": host_offload,
            "layer_swap_latency_ms": swap_latency_ms,
            "is_offload_required": is_offload,
            "total_model_size_mb": total_model_mb,
        }
