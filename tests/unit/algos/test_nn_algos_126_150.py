import pytest
import math

from src.features.code_engine.algos.nn.mixture_of_experts.moe_load_balancing.impl import (
    NnAlgoMoeLoadBalancing,
)
from src.features.code_engine.algos.nn.mixture_of_experts.moe_expert_choice_routing.impl import (
    NnAlgoMoeExpertChoiceRouting,
)
from src.features.code_engine.algos.nn.multimodal_memory.vision_transformer.impl import (
    NnAlgoVisionTransformer,
)
from src.features.code_engine.algos.nn.multimodal_memory.swin_transformer.impl import (
    NnAlgoSwinTransformer,
)
from src.features.code_engine.algos.nn.multimodal_memory.perceiver_latent_attention.impl import (
    NnAlgoPerceiverLatentAttention,
)
from src.features.code_engine.algos.nn.multimodal_memory.cross_attention_conditioning.impl import (
    NnAlgoCrossAttentionConditioning,
)
from src.features.code_engine.algos.nn.multimodal_memory.clip_contrastive_pretraining.impl import (
    NnAlgoClipContrastivePretraining,
)
from src.features.code_engine.algos.nn.multimodal_memory.multimodal_llm_connectors.impl import (
    NnAlgoMultimodalLlmConnectors,
)
from src.features.code_engine.algos.nn.multimodal_memory.segment_level_recurrence.impl import (
    NnAlgoSegmentLevelRecurrence,
)
from src.features.code_engine.algos.nn.multimodal_memory.retrieval_enhanced_transformers.impl import (
    NnAlgoRetrievalEnhancedTransformers,
)
from src.features.code_engine.algos.nn.multimodal_memory.prefix_lm_mixture_denoisers.impl import (
    NnAlgoPrefixLmMixtureDenoisers,
)
from src.features.code_engine.algos.nn.multimodal_memory.multi_token_prediction.impl import (
    NnAlgoMultiTokenPrediction,
)
from src.features.code_engine.algos.nn.scaling_stability.scaling_laws_compute_optimal.impl import (
    NnAlgoScalingLawsComputeOptimal,
)
from src.features.code_engine.algos.nn.scaling_stability.attention_sinks_streaming.impl import (
    NnAlgoAttentionSinksStreaming,
)
from src.features.code_engine.algos.nn.scaling_stability.logit_stabilization_z_loss.impl import (
    NnAlgoLogitStabilizationZLoss,
)
from src.features.code_engine.algos.nn.decoding.greedy_temperature_sampling.impl import (
    NnAlgoGreedyTemperatureSampling,
)
from src.features.code_engine.algos.nn.decoding.top_k_sampling.impl import (
    NnAlgoTopKSampling,
)
from src.features.code_engine.algos.nn.decoding.nucleus_min_p_sampling.impl import (
    NnAlgoNucleusMinPSampling,
)
from src.features.code_engine.algos.nn.decoding.repetition_frequency_penalties.impl import (
    NnAlgoRepetitionFrequencyPenalties,
)
from src.features.code_engine.algos.nn.decoding.contrastive_search_decoding.impl import (
    NnAlgoContrastiveSearchDecoding,
)
from src.features.code_engine.algos.nn.decoding.speculative_decoding.impl import (
    NnAlgoSpeculativeDecoding,
)
from src.features.code_engine.algos.nn.decoding.self_drafting_medusa_eagle.impl import (
    NnAlgoSelfDraftingMedusaEagle,
)
from src.features.code_engine.algos.nn.decoding.constrained_grammar_decoding.impl import (
    NnAlgoConstrainedGrammarDecoding,
)
from src.features.code_engine.algos.nn.decoding.self_consistency_sampling.impl import (
    NnAlgoSelfConsistencySampling,
)
from src.features.code_engine.algos.nn.decoding.lookahead_jacobi_decoding.impl import (
    NnAlgoLookaheadJacobiDecoding,
)


def test_algo_nn_126_moe_load_balancing():
    probs = [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]]
    res = NnAlgoMoeLoadBalancing.compute_loss_and_capacity(probs, capacity_factor=1.5, num_experts=2)
    assert res["aux_loss"] >= 0.0
    assert 0.0 <= res["token_drop_rate"] <= 1.0


def test_algo_nn_127_moe_expert_choice_routing():
    affinity = [[2.0, 1.0], [0.5, 3.0], [1.5, 2.5]]
    res = NnAlgoMoeExpertChoiceRouting.route(affinity, tokens_per_expert=2)
    assert len(res["expert_assignments"]) == 2
    assert res["load_variance"] == 0.0


def test_algo_nn_128_vision_transformer():
    img = [[0.1] * 32 for _ in range(32)]
    res = NnAlgoVisionTransformer.patchify_and_embed(img, patch_size=16, embed_dim=32)
    assert res["num_patches"] == 4
    assert len(res["patch_tokens"]) == 5


def test_algo_nn_129_swin_transformer():
    grid = [[[0.1, 0.2] for _ in range(14)] for _ in range(14)]
    res = NnAlgoSwinTransformer.partition_windows(grid, window_size=7, shift_size=0)
    assert res["num_windows"] == 4
    assert res["window_tokens"] == 49


def test_algo_nn_130_perceiver_latent_attention():
    inputs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    latents = [[0.0, 0.0], [1.0, 1.0]]
    res = NnAlgoPerceiverLatentAttention.forward(inputs, latents)
    assert len(res["updated_latents"]) == 2
    assert res["compression_ratio"] == 2.0


def test_algo_nn_131_cross_attention_conditioning():
    primary = [[0.1, 0.2], [0.3, 0.4]]
    context = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    res = NnAlgoCrossAttentionConditioning.forward(primary, context)
    assert len(res["conditioned_output"]) == 2
    assert len(res["conditioned_output"][0]) == 2


def test_algo_nn_132_clip_contrastive_pretraining():
    img_emb = [[1.0, 0.0], [0.0, 1.0]]
    txt_emb = [[1.0, 0.0], [0.0, 1.0]]
    res = NnAlgoClipContrastivePretraining.forward(img_emb, txt_emb, logit_scale=1.0)
    assert res["loss"] >= 0.0
    assert res["accuracy"] == 1.0


def test_algo_nn_133_multimodal_llm_connectors():
    vis = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    res = NnAlgoMultimodalLlmConnectors.project(vis, llm_embed_dim=8)
    assert res["token_count"] == 2
    assert len(res["projected_tokens"][0]) == 8


def test_algo_nn_134_segment_level_recurrence():
    curr = [[0.1, 0.2], [0.3, 0.4]]
    mem = [[0.0, 0.1]]
    res = NnAlgoSegmentLevelRecurrence.forward(curr, mem)
    assert res["total_context_length"] == 3
    assert len(res["updated_memory"]) == 2


def test_algo_nn_135_retrieval_enhanced_transformers():
    in_chunks = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
    ret_chunks = in_chunks
    res = NnAlgoRetrievalEnhancedTransformers.forward(in_chunks, ret_chunks)
    assert res["num_chunks"] == 2


def test_algo_nn_136_prefix_lm_mixture_denoisers():
    res = NnAlgoPrefixLmMixtureDenoisers.construct_mask(seq_len=6, prefix_length=3, mode="S_denoiser")
    assert len(res["attention_mask"]) == 6
    assert res["attention_mask"][0][2] == 0.0
    assert res["attention_mask"][3][4] == -1e9


def test_algo_nn_137_multi_token_prediction():
    trunk = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    targets = [1, 2, 3, 4, 5]
    res = NnAlgoMultiTokenPrediction.forward(trunk, targets, num_future_tokens=2)
    assert res["composite_loss"] >= 0.0
    assert len(res["per_head_losses"]) == 2


def test_algo_nn_138_scaling_laws_compute_optimal():
    res = NnAlgoScalingLawsComputeOptimal.calculate_budget(compute_budget_flops=6e18)
    assert res["optimal_parameters"] > 0.0
    assert res["optimal_training_tokens"] > 0.0
    assert res["estimated_loss"] > 0.0


def test_algo_nn_139_attention_sinks_streaming():
    stream = list(range(100))
    res = NnAlgoAttentionSinksStreaming.filter_stream(stream, num_sink_tokens=4, window_size=16)
    assert len(res["retained_cache_tokens"]) == 20
    assert res["retained_cache_tokens"][:4] == [0, 1, 2, 3]


def test_algo_nn_140_logit_stabilization_z_loss():
    logits = [[10.0, 5.0, 2.0], [8.0, 7.0, 6.0]]
    res = NnAlgoLogitStabilizationZLoss.compute(logits, alpha=1e-4)
    assert res["z_loss"] > 0.0
    assert res["max_logit"] == 10.0


def test_algo_nn_141_greedy_temperature_sampling():
    logits = [1.0, 5.0, 2.0]
    res_greedy = NnAlgoGreedyTemperatureSampling.sample(logits, temperature=0.0)
    assert res_greedy["selected_token"] == 1
    assert res_greedy["is_greedy"] is True

    res_temp = NnAlgoGreedyTemperatureSampling.sample(logits, temperature=0.8)
    assert res_temp["selected_token"] == 1


def test_algo_nn_142_top_k_sampling():
    logits = [0.1, 2.0, 5.0, 1.2, 0.3]
    res = NnAlgoTopKSampling.sample(logits, k=2)
    assert len(res["retained_indices"]) == 2
    assert res["retained_indices"][0] == 2
    assert res["selected_token"] == 2


def test_algo_nn_143_nucleus_min_p_sampling():
    logits = [5.0, 4.0, 0.1, 0.01]
    res = NnAlgoNucleusMinPSampling.sample(logits, top_p=0.9, min_p=0.05)
    assert len(res["retained_indices"]) >= 1
    assert res["selected_token"] == 0


def test_algo_nn_144_repetition_frequency_penalties():
    logits = [2.0, 2.0, 2.0]
    past = [0, 0, 1]
    res = NnAlgoRepetitionFrequencyPenalties.apply(logits, past, repetition_penalty=2.0, frequency_penalty=0.5)
    assert res["adjusted_logits"][0] < res["adjusted_logits"][2]
    assert res["penalized_tokens_count"] == 2


def test_algo_nn_145_contrastive_search_decoding():
    cand_probs = [0.6, 0.4]
    cand_h = [[1.0, 0.0], [0.0, 1.0]]
    past_h = [[1.0, 0.0]]
    res = NnAlgoContrastiveSearchDecoding.select_candidate(cand_probs, cand_h, past_h, alpha=0.8)
    assert res["best_candidate_index"] == 1


def test_algo_nn_146_speculative_decoding():
    draft_tokens = [10, 20, 30]
    draft_probs = [0.9, 0.9, 0.9]
    target_probs = [0.95, 0.95, 0.95]
    res = NnAlgoSpeculativeDecoding.verify(draft_tokens, draft_probs, target_probs, bonus_token=40)
    assert res["acceptance_count"] == 4
    assert res["accepted_tokens"] == [10, 20, 30, 40]


def test_algo_nn_147_self_drafting_medusa_eagle():
    candidates = [[10, 11], [20, 21]]
    res = NnAlgoSelfDraftingMedusaEagle.build_tree(candidates)
    assert res["total_nodes"] == 4
    assert len(res["tree_paths"]) == 4


def test_algo_nn_148_constrained_grammar_decoding():
    logits = [1.0, 5.0, 10.0, 2.0]
    allowed = [0, 1, 3]
    res = NnAlgoConstrainedGrammarDecoding.apply_mask(logits, allowed)
    assert res["selected_token"] == 1
    assert res["valid_token_count"] == 3


def test_algo_nn_149_self_consistency_sampling():
    answers = ["42", "42", "24", "42"]
    res = NnAlgoSelfConsistencySampling.aggregate_votes(answers)
    assert res["consensus_answer"] == "42"
    assert res["agreement_rate"] == 0.75


def test_algo_nn_150_lookahead_jacobi_decoding():
    guesses = [1, 2, 3, 4]
    res = NnAlgoLookaheadJacobiDecoding.iterate(guesses, max_iterations=5)
    assert res["num_tokens_stabilized"] == 4
    assert len(res["stabilized_tokens"]) == 4
