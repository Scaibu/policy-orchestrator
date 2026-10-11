import pytest
from src.features.code_engine.algos.nn.training_at_scale.long_context_training.impl import (
    NnAlgoLongContextTraining,
)
from src.features.code_engine.algos.nn.training_at_scale.sequence_packing_doc_masking.impl import (
    NnAlgoSequencePackingDocMasking,
)
from src.features.code_engine.algos.nn.training_at_scale.mixed_precision_training.impl import (
    NnAlgoMixedPrecisionTraining,
)
from src.features.code_engine.algos.nn.training_at_scale.fp8_training.impl import (
    NnAlgoFp8Training,
)
from src.features.code_engine.algos.nn.training_at_scale.data_parallelism_allreduce.impl import (
    NnAlgoDataParallelismAllreduce,
)
from src.features.code_engine.algos.nn.training_at_scale.zero_fsdp_sharded_training.impl import (
    NnAlgoZeroFsdpShardedTraining,
)
from src.features.code_engine.algos.nn.training_at_scale.tensor_parallelism_megatron.impl import (
    NnAlgoTensorParallelismMegatron,
)
from src.features.code_engine.algos.nn.training_at_scale.pipeline_parallelism_1f1b.impl import (
    NnAlgoPipelineParallelism1f1b,
)
from src.features.code_engine.algos.nn.training_at_scale.context_parallelism_ring_attention.impl import (
    NnAlgoContextParallelismRingAttention,
)
from src.features.code_engine.algos.nn.training_at_scale.expert_parallelism_alltoall.impl import (
    NnAlgoExpertParallelismAlltoall,
)
from src.features.code_engine.algos.nn.training_at_scale.gradient_accumulation.impl import (
    NnAlgoGradientAccumulation,
)
from src.features.code_engine.algos.nn.training_at_scale.offloading_activation_parameter.impl import (
    NnAlgoOffloadingActivationParameter,
)
from src.features.code_engine.algos.nn.training_at_scale.distributed_checkpointing_resumption.impl import (
    NnAlgoDistributedCheckpointingResumption,
)
from src.features.code_engine.algos.nn.training_at_scale.loss_spike_detection_recovery.impl import (
    NnAlgoLossSpikeDetectionRecovery,
)
from src.features.code_engine.algos.nn.training_at_scale.data_mixture_optimization_doremi.impl import (
    NnAlgoDataMixtureOptimizationDoremi,
)
from src.features.code_engine.algos.nn.training_at_scale.critical_batch_size_noise_scale.impl import (
    NnAlgoCriticalBatchSizeNoiseScale,
)
from src.features.code_engine.algos.nn.training_at_scale.synthetic_data_generation_filtering.impl import (
    NnAlgoSyntheticDataGenerationFiltering,
)
from src.features.code_engine.algos.nn.training_at_scale.federated_learning_fedavg.impl import (
    NnAlgoFederatedLearningFedavg,
)
from src.features.code_engine.algos.nn.training_at_scale.differentially_private_training_dpsgd.impl import (
    NnAlgoDifferentiallyPrivateTrainingDpsgd,
)


def test_algo_229_long_context_training():
    res = NnAlgoLongContextTraining.forward(
        current_step=2500,
        total_steps=10000,
        stage_boundaries=[2000, 5000],
        stage_sequence_lengths=[2048, 8192, 32768],
    )
    assert res["current_stage"] == 1
    assert res["active_sequence_length"] == 8192
    assert res["active_rope_theta"] > 10000.0
    assert 0.0 <= res["stage_progress_fraction"] <= 1.0


def test_algo_230_sequence_packing_doc_masking():
    doc_lens = [300, 500, 200, 700, 400]
    res = NnAlgoSequencePackingDocMasking.forward(
        document_lengths=doc_lens,
        max_sequence_length=1000,
    )
    assert res["total_packed_sequences"] >= 2
    assert res["padding_efficiency"] > 0.5
    for b_len in res["bin_lengths"]:
        assert b_len <= 1000


def test_algo_231_mixed_precision_training():
    # Normal case without overflow
    res_ok = NnAlgoMixedPrecisionTraining.forward(
        raw_gradients=[655.36, 1310.72],
        loss_scale=65536.0,
        precision_mode="fp16_dynamic",
    )
    assert not res_ok["has_overflow"]
    assert not res_ok["step_skipped"]
    assert abs(res_ok["unscaled_gradients"][0] - 0.01) < 1e-5

    # Overflow case
    res_nan = NnAlgoMixedPrecisionTraining.forward(
        raw_gradients=[float("nan"), 100.0],
        loss_scale=65536.0,
    )
    assert res_nan["has_overflow"]
    assert res_nan["step_skipped"]
    assert res_nan["updated_loss_scale"] == 32768.0


def test_algo_232_fp8_training():
    vals = [0.1, -0.5, 1.2, -3.4, 0.0]
    res = NnAlgoFp8Training.forward(
        tensor_values=vals,
        fp8_format="e4m3",
        current_scale=10.0,
        scaling_mode="current",
    )
    assert len(res["quantized_integers"]) == 5
    assert len(res["dequantized_values"]) == 5
    assert res["quantization_snr_db"] > 10.0


def test_algo_233_data_parallelism_allreduce():
    worker_grads = [
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
    ]
    res = NnAlgoDataParallelismAllreduce.forward(worker_gradients=worker_grads)
    assert res["averaged_gradients"] == [2.0, 3.0, 4.0]
    assert res["worker_count"] == 2
    assert res["communication_steps"] == 2


def test_algo_234_zero_fsdp():
    res = NnAlgoZeroFsdpShardedTraining.forward(
        total_parameters=1000000,
        num_devices=8,
        zero_stage=3,
    )
    assert res["memory_saving_factor"] > 4.0
    assert res["sharded_parameter_count_per_device"] == 125000


def test_algo_235_tensor_parallelism_megatron():
    x = [1.0, 0.5]  # d_in = 2
    # TP = 2, d_mid = 4 (2 per shard), d_out = 2
    col_shards = [
        [[0.5, 0.5], [0.2, 0.1]],  # shard 0: (2, 2)
        [[0.1, 0.3], [0.4, 0.4]],  # shard 1: (2, 2)
    ]
    row_shards = [
        [[1.0, 0.5], [0.2, 0.8]],  # shard 0: (2, 2)
        [[0.5, 1.0], [0.8, 0.2]],  # shard 1: (2, 2)
    ]
    res = NnAlgoTensorParallelismMegatron.forward(
        input_vector=x,
        column_shards=col_shards,
        row_shards=row_shards,
    )
    assert len(res["output_vector"]) == 2
    assert res["tp_degree"] == 2


def test_algo_236_pipeline_parallelism_1f1b():
    res = NnAlgoPipelineParallelism1f1b.forward(
        num_pipeline_stages=4,
        num_microbatches=16,
    )
    assert res["pipeline_bubble_fraction"] < 0.25
    assert res["max_stored_activation_microbatches"] == 4


def test_algo_237_context_parallelism_ring_attention():
    q = [[1.0, 0.0], [0.0, 1.0]]  # C=2, d=2
    k_ring = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.5, 0.5], [0.0, 0.5]],
    ]
    v_ring = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]
    res = NnAlgoContextParallelismRingAttention.forward(
        local_queries=q,
        ring_keys=k_ring,
        ring_values=v_ring,
    )
    assert len(res["context_attention_output"]) == 2
    assert len(res["context_attention_output"][0]) == 2
    assert res["total_ring_rounds"] == 2


def test_algo_238_expert_parallelism_alltoall():
    token_experts = [
        [0, 1],
        [2, 3],
        [0, 2],
    ]
    exp_map = {"0": 0, "1": 1, "2": 0, "3": 1}
    res = NnAlgoExpertParallelismAlltoall.forward(
        token_expert_indices=token_experts,
        expert_to_rank_map=exp_map,
        num_ranks=2,
    )
    assert sum(res["rank_dispatch_counts"]) == 6
    assert res["load_imbalance_ratio"] >= 1.0


def test_algo_239_gradient_accumulation():
    acc = [0.0, 0.0]
    mb1 = [2.0, 4.0]
    mb2 = [4.0, 6.0]

    res1 = NnAlgoGradientAccumulation.forward(
        accumulated_gradients=acc,
        microbatch_gradients=mb1,
        current_microbatch=1,
        total_accumulation_steps=2,
    )
    assert not res1["is_update_step"]
    assert res1["updated_accumulated_gradients"] == [2.0, 4.0]

    res2 = NnAlgoGradientAccumulation.forward(
        accumulated_gradients=res1["updated_accumulated_gradients"],
        microbatch_gradients=mb2,
        current_microbatch=2,
        total_accumulation_steps=2,
    )
    assert res2["is_update_step"]
    assert res2["normalized_gradients_for_step"] == [3.0, 5.0]


def test_algo_240_offloading_activation_parameter():
    res = NnAlgoOffloadingActivationParameter.forward(
        gpu_memory_capacity_mb=4000.0,
        total_layers=32,
        layer_parameter_mb=500.0,
        layer_activation_mb=200.0,
    )
    assert res["is_offload_required"]
    assert res["gpu_resident_memory_mb"] == 700.0
    assert res["layer_swap_latency_ms"] > 0.0


def test_algo_241_distributed_checkpointing_resumption():
    shards = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    res = NnAlgoDistributedCheckpointingResumption.forward(
        sharded_weights=shards,
        global_step=1000,
        data_loader_offset=50000,
        target_num_shards=3,
    )
    assert len(res["resharded_weights"]) == 3
    assert res["total_checkpoint_elements"] == 6
    assert len(res["sha256_checksum"]) == 64


def test_algo_242_loss_spike_detection_recovery():
    history = [1.0, 1.1, 0.95, 1.05, 0.98, 1.02] * 4  # std approx 0.05
    # Normal loss
    res_normal = NnAlgoLossSpikeDetectionRecovery.forward(
        current_loss=1.02,
        current_grad_norm=0.5,
        loss_history=history,
    )
    assert not res_normal["is_spike_detected"]
    assert res_normal["recommended_action"] == "proceed"

    # Spike loss
    res_spike = NnAlgoLossSpikeDetectionRecovery.forward(
        current_loss=10.0,
        current_grad_norm=15.0,
        loss_history=history,
    )
    assert res_spike["is_spike_detected"]
    assert res_spike["recommended_action"] in ["rollback_and_skip", "throttle_lr"]


def test_algo_243_data_mixture_optimization_doremi():
    proxy = [2.5, 1.8, 3.0]
    ref = [2.0, 1.5, 2.0]  # Excess losses: [0.5, 0.3, 1.0]
    weights = [0.33, 0.33, 0.34]

    res = NnAlgoDataMixtureOptimizationDoremi.forward(
        domain_losses_proxy=proxy,
        domain_losses_reference=ref,
        current_domain_weights=weights,
        step_size_eta=0.5,
    )
    assert res["worst_case_domain_idx"] == 2
    assert res["updated_domain_weights"][2] > res["updated_domain_weights"][1]
    assert abs(sum(res["updated_domain_weights"]) - 1.0) < 1e-5


def test_algo_244_critical_batch_size():
    sample_grads = [
        [1.0, 0.5],
        [1.2, 0.4],
        [0.8, 0.6],
        [1.1, 0.5],
    ]
    res = NnAlgoCriticalBatchSizeNoiseScale.forward(
        per_sample_gradients=sample_grads,
        current_step=500,
        total_steps=1000,
        min_batch_size=16,
        max_batch_size=512,
    )
    assert res["estimated_noise_scale_b_noise"] > 0.0
    assert 16 <= res["recommended_batch_size"] <= 512


def test_algo_245_synthetic_data_filtering():
    samples = [
        {"id": 1, "text": "This is a valid detailed high-quality response for coding python.", "quality_score": 0.9, "token_count": 12},
        {"id": 2, "text": "Too short", "quality_score": 0.95, "token_count": 2},
        {"id": 3, "text": "Low score garbage text for testing filters thoroughly.", "quality_score": 0.4, "token_count": 10},
        {"id": 4, "text": "This is a valid detailed high-quality response for coding python duplicate.", "quality_score": 0.9, "token_count": 12},
    ]
    res = NnAlgoSyntheticDataGenerationFiltering.forward(
        candidate_samples=samples,
        min_quality_score=0.7,
        min_token_length=5,
        jaccard_dedup_threshold=0.8,
    )
    assert len(res["accepted_samples"]) >= 1
    assert res["rejected_count"] >= 2


def test_algo_246_federated_learning_fedavg():
    updates = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    counts = [100, 300]
    global_w = [1.0, 1.0]
    res = NnAlgoFederatedLearningFedavg.forward(
        client_model_updates=updates,
        client_sample_counts=counts,
        global_weights=global_w,
    )
    # Expected delta = 0.25 * [0.1, 0.2] + 0.75 * [0.3, 0.4] = [0.025 + 0.225, 0.05 + 0.3] = [0.25, 0.35]
    assert abs(res["aggregated_delta"][0] - 0.25) < 1e-6
    assert abs(res["updated_global_weights"][0] - 1.25) < 1e-6


def test_algo_247_dpsgd():
    grads = [
        [1.0, 2.0],
        [0.5, 0.5],
        [3.0, 4.0],
    ]
    res = NnAlgoDifferentiallyPrivateTrainingDpsgd.forward(
        per_example_gradients=grads,
        clipping_norm_c=1.0,
        noise_multiplier_sigma=1.0,
    )
    assert len(res["private_averaged_gradients"]) == 2
    assert res["mean_clipped_norm"] <= 1.0
    assert res["estimated_epsilon"] > 0.0
