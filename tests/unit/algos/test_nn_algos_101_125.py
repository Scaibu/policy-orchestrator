import pytest
import math

from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)
from src.features.code_engine.algos.nn.attention_objectives.multi_head_attention.impl import (
    NnAlgoMultiHeadAttention,
)
from src.features.code_engine.algos.nn.attention_objectives.causal_language_modeling.impl import (
    NnAlgoCausalLanguageModeling,
)
from src.features.code_engine.algos.nn.attention_objectives.masked_language_modeling.impl import (
    NnAlgoMaskedLanguageModeling,
)
from src.features.code_engine.algos.nn.attention_objectives.span_corruption.impl import (
    NnAlgoSpanCorruption,
)
from src.features.code_engine.algos.nn.position_encodings.sinusoidal_positional_encoding.impl import (
    NnAlgoSinusoidalPositionalEncoding,
)
from src.features.code_engine.algos.nn.position_encodings.learned_absolute_position_embeddings.impl import (
    NnAlgoLearnedAbsolutePositionEmbeddings,
)
from src.features.code_engine.algos.nn.position_encodings.rotary_position_embeddings.impl import (
    NnAlgoRotaryPositionEmbeddings,
)
from src.features.code_engine.algos.nn.position_encodings.alibi_attention_linear_biases.impl import (
    NnAlgoAlibiAttentionLinearBiases,
)
from src.features.code_engine.algos.nn.position_encodings.relative_position_bias.impl import (
    NnAlgoRelativePositionBias,
)
from src.features.code_engine.algos.nn.position_encodings.context_window_extension.impl import (
    NnAlgoContextWindowExtension,
)
from src.features.code_engine.algos.nn.transformer_efficiency.transformer_block.impl import (
    NnAlgoTransformerBlock,
)
from src.features.code_engine.algos.nn.transformer_efficiency.encoder_decoder_transformers.impl import (
    NnAlgoEncoderDecoderTransformers,
)
from src.features.code_engine.algos.nn.transformer_efficiency.grouped_query_attention.impl import (
    NnAlgoGroupedQueryAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.multi_head_latent_attention.impl import (
    NnAlgoMultiHeadLatentAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.flash_attention.impl import (
    NnAlgoFlashAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.kv_cache.impl import (
    NnAlgoKvCache,
)
from src.features.code_engine.algos.nn.transformer_efficiency.paged_attention.impl import (
    NnAlgoPagedAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.sliding_window_attention.impl import (
    NnAlgoSlidingWindowAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.sparse_attention_patterns.impl import (
    NnAlgoSparseAttentionPatterns,
)
from src.features.code_engine.algos.nn.transformer_efficiency.linear_attention.impl import (
    NnAlgoLinearAttention,
)
from src.features.code_engine.algos.nn.transformer_efficiency.structured_state_space_s4.impl import (
    NnAlgoStructuredStateSpaceS4,
)
from src.features.code_engine.algos.nn.transformer_efficiency.mamba_selective_ssm.impl import (
    NnAlgoMambaSelectiveSsm,
)
from src.features.code_engine.algos.nn.transformer_efficiency.rwkv_retnet_recurrent.impl import (
    NnAlgoRwkvRetnetRecurrent,
)
from src.features.code_engine.algos.nn.mixture_of_experts.moe_top_k_gating.impl import (
    NnAlgoMoeTopKGating,
)


def test_algo_nn_101_scaled_dot_product_attention():
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[1.0, 2.0], [3.0, 4.0]]
    res = NnAlgoScaledDotProductAttention.forward(q, k, v)
    assert len(res["output"]) == 2
    assert len(res["attention_weights"]) == 2
    assert math.isclose(sum(res["attention_weights"][0]), 1.0, rel_tol=1e-5)


def test_algo_nn_102_multi_head_attention():
    q = [[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.1, 0.8]]
    k = q
    v = q
    res = NnAlgoMultiHeadAttention.forward(q, k, v, num_heads=2)
    assert len(res["output"]) == 2
    assert len(res["output"][0]) == 4
    assert res["head_dim"] == 2


def test_algo_nn_103_causal_language_modeling():
    logits = [[2.0, 0.1, 0.5], [0.2, 1.8, 0.3]]
    targets = [0, 1]
    res = NnAlgoCausalLanguageModeling.forward(logits, targets)
    assert res["loss"] > 0.0
    assert res["perplexity"] >= 1.0
    assert res["num_active_tokens"] == 2


def test_algo_nn_104_masked_language_modeling():
    logits = [[2.0, 0.1], [0.1, 2.0], [1.5, 0.2]]
    masked_pos = [0, 1]
    targets = [0, 1]
    res = NnAlgoMaskedLanguageModeling.forward(logits, masked_pos, targets)
    assert res["accuracy"] == 1.0
    assert res["loss"] > 0.0


def test_algo_nn_105_span_corruption():
    tokens = [1, 2, 3, 4, 5, 6, 7]
    spans = [(2, 4)]
    res = NnAlgoSpanCorruption.forward(tokens, spans, sentinel_start_id=32000)
    assert 32000 in res["corrupted_inputs"]
    assert res["num_spans_corrupted"] == 1


def test_algo_nn_106_sinusoidal_positional_encoding():
    res = NnAlgoSinusoidalPositionalEncoding.forward(seq_len=8, d_model=16)
    assert len(res["encoding_matrix"]) == 8
    assert len(res["encoding_matrix"][0]) == 16


def test_algo_nn_107_learned_absolute_position_embeddings():
    tok_emb = [[0.5, 0.2], [0.1, 0.9]]
    pos_table = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
    res = NnAlgoLearnedAbsolutePositionEmbeddings.forward(tok_emb, pos_table, offset=0)
    assert len(res["output_embeddings"]) == 2
    assert math.isclose(res["output_embeddings"][0][0], 0.51, rel_tol=1e-5)


def test_algo_nn_108_rotary_position_embeddings():
    vecs = [[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.2, 0.8]]
    pos = [0, 1]
    res = NnAlgoRotaryPositionEmbeddings.forward(vecs, pos)
    assert len(res["rotated_vectors"]) == 2
    assert res["dim"] == 4


def test_algo_nn_109_alibi_attention_linear_biases():
    res = NnAlgoAlibiAttentionLinearBiases.forward(num_heads=4, seq_len_q=4, seq_len_k=4)
    assert len(res["bias_matrices"]) == 4
    assert res["bias_matrices"][0][0][0] == 0.0
    assert res["bias_matrices"][0][0][1] < 0.0


def test_algo_nn_110_relative_position_bias():
    res = NnAlgoRelativePositionBias.forward(seq_len_q=4, seq_len_k=4, num_buckets=16)
    assert len(res["bucket_matrix"]) == 4
    assert len(res["bucket_matrix"][0]) == 4


def test_algo_nn_111_context_window_extension():
    res = NnAlgoContextWindowExtension.forward(scale_factor=2.0, method="linear_interpolation", original_max_len=2048)
    assert res["extended_max_len"] == 4096
    assert len(res["scaled_frequencies"]) == 64


def test_algo_nn_112_transformer_block():
    hidden = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    res = NnAlgoTransformerBlock.forward(hidden, num_heads=2, d_ff=8)
    assert len(res["output_states"]) == 2
    assert len(res["output_states"][0]) == 4


def test_algo_nn_113_encoder_decoder_transformers():
    enc = [[0.1, 0.2], [0.3, 0.4]]
    dec = [[0.5, 0.6]]
    res = NnAlgoEncoderDecoderTransformers.forward(enc, dec)
    assert len(res["cross_attention_output"]) == 1
    assert len(res["cross_attention_output"][0]) == 2


def test_algo_nn_114_grouped_query_attention():
    q = [[0.1, 0.2, 0.3, 0.4]]
    k = [[0.1, 0.2]]
    v = [[0.5, 0.6]]
    res = NnAlgoGroupedQueryAttention.forward(q, k, v, num_q_heads=2, num_kv_heads=1)
    assert len(res["output"]) == 1
    assert res["compression_ratio"] == 2.0


def test_algo_nn_115_multi_head_latent_attention():
    hidden = [[0.1, 0.2, 0.3, 0.4]]
    res = NnAlgoMultiHeadLatentAttention.forward(hidden, latent_dim=2, num_heads=2, head_dim=2)
    assert len(res["compressed_kv_latent"]) == 1
    assert res["compression_factor"] == 4.0


def test_algo_nn_116_flash_attention():
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[1.0, 2.0], [3.0, 4.0]]
    res = NnAlgoFlashAttention.forward(q, k, v, block_size=1)
    assert len(res["output"]) == 2
    assert len(res["output"][0]) == 2


def test_algo_nn_117_kv_cache():
    res = NnAlgoKvCache.append_step([0.1, 0.2], [0.3, 0.4])
    assert res["sequence_length"] == 1
    res2 = NnAlgoKvCache.append_step([0.5, 0.6], [0.7, 0.8], res["updated_keys"], res["updated_values"])
    assert res2["sequence_length"] == 2


def test_algo_nn_118_paged_attention():
    table = [0, 1]
    pool = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
    q = [0.1, 0.2]
    res = NnAlgoPagedAttention.forward(table, pool, q, block_size=2)
    assert res["total_tokens"] == 4
    assert len(res["gathered_keys"]) == 4


def test_algo_nn_119_sliding_window_attention():
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k = q
    v = q
    res = NnAlgoSlidingWindowAttention.forward(q, k, v, window_size=1)
    assert len(res["output"]) == 3
    assert res["active_window_size"] == 1


def test_algo_nn_120_sparse_attention_patterns():
    res = NnAlgoSparseAttentionPatterns.construct_pattern(seq_len=8, window_radius=1, global_indices=[0])
    assert len(res["adjacency_mask"]) == 8
    assert res["total_edges"] > 0
    assert 0.0 <= res["sparsity_ratio"] <= 1.0


def test_algo_nn_121_linear_attention():
    q = [[0.1, 0.2], [0.3, 0.4]]
    k = q
    v = [[1.0, 2.0], [3.0, 4.0]]
    res = NnAlgoLinearAttention.forward(q, k, v)
    assert len(res["output"]) == 2
    assert len(res["output"][0]) == 2


def test_algo_nn_122_structured_state_space_s4():
    u = [1.0, 0.5, -0.2, 0.8]
    res = NnAlgoStructuredStateSpaceS4.forward(u, state_dim=4, delta=0.01)
    assert len(res["outputs"]) == 4
    assert len(res["final_state"]) == 4


def test_algo_nn_123_mamba_selective_ssm():
    u = [0.5, 1.2, -0.4, 0.1]
    res = NnAlgoMambaSelectiveSsm.forward(u, state_dim=4)
    assert len(res["outputs"]) == 4
    assert len(res["final_state"]) == 4


def test_algo_nn_124_rwkv_retnet_recurrent():
    q = [[0.1, 0.2], [0.3, 0.4]]
    k = q
    v = [[1.0, 2.0], [3.0, 4.0]]
    res = NnAlgoRwkvRetnetRecurrent.forward(q, k, v, gamma=0.9)
    assert len(res["output"]) == 2
    assert len(res["final_recurrent_state"]) == 2


def test_algo_nn_125_moe_top_k_gating():
    logits = [[2.0, 1.0, 0.5, 0.1], [0.2, 1.8, 3.0, 0.1]]
    res = NnAlgoMoeTopKGating.route(logits, top_k=2)
    assert len(res["selected_experts"]) == 2
    assert len(res["selected_experts"][0]) == 2
    assert math.isclose(sum(res["gating_weights"][0]), 1.0, rel_tol=1e-5)
