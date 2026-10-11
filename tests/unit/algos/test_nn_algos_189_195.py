import pytest
import math

from src.features.code_engine.algos.nn.timeseries_recommendation.basis_expansion_nbeats.impl import (
    NnAlgoBasisExpansionNbeats,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.patch_time_series_transformer.impl import (
    NnAlgoPatchTimeSeriesTransformer,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.deep_cross_network_v2.impl import (
    NnAlgoDeepCrossNetworkV2,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.sequential_recommendation_sasrec.impl import (
    NnAlgoSequentialRecommendationSasrec,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.multi_gate_mixture_of_experts_mmoe.impl import (
    NnAlgoMultiGateMixtureOfExpertsMmoe,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.deep_interest_network_attention.impl import (
    NnAlgoDeepInterestNetworkAttention,
)
from src.features.code_engine.algos.nn.timeseries_recommendation.deep_learning_recommendation_dlrm.impl import (
    NnAlgoDeepLearningRecommendationDlrm,
)


def test_algo_189_nbeats():
    lookback = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights_b = [[0.1] * 5, [0.05] * 5]
    weights_f = [[0.2] * 5, [0.1] * 5]
    res = NnAlgoBasisExpansionNbeats.forward(
        lookback_window=lookback,
        weights_theta_backcast=weights_b,
        weights_theta_forecast=weights_f,
        forecast_horizon=3,
        basis_type="polynomial",
    )
    assert len(res["backcast"]) == 5
    assert len(res["forecast"]) == 3
    assert len(res["residual_lookback"]) == 5
    assert res["forecast_energy"] >= 0.0
    for i in range(5):
        assert abs(res["residual_lookback"][i] - (lookback[i] - res["backcast"][i])) < 1e-6


def test_algo_190_patchtst():
    ts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    weights = [[0.5, 0.5], [0.2, -0.2], [0.1, 0.3]]
    res = NnAlgoPatchTimeSeriesTransformer.forward(
        time_series=ts,
        patch_len=2,
        stride=2,
        projection_weights=weights,
    )
    assert len(res["normalized_series"]) == 8
    assert len(res["patch_tokens"]) == 4
    assert len(res["projected_embeddings"]) == 4
    assert res["instance_std"] > 0.0


def test_algo_191_dcn_v2():
    x0 = [1.0, -1.0]
    w_cross = [[0.5, 0.1], [0.2, 0.4]]
    b_cross = [0.1, -0.1]
    w_deep = [[0.3, 0.2], [-0.1, 0.5], [0.4, 0.1]]
    b_deep = [0.0, 0.1, -0.1]
    w_out = [0.1] * (2 + 3)
    res = NnAlgoDeepCrossNetworkV2.forward(
        input_features_x0=x0,
        cross_weights_w=w_cross,
        cross_bias_b=b_cross,
        deep_weights_w=w_deep,
        deep_bias_b=b_deep,
        output_weights=w_out,
    )
    assert len(res["cross_output"]) == 2
    assert len(res["deep_output"]) == 3
    assert 0.0 <= res["predicted_probability"] <= 1.0


def test_algo_192_sasrec():
    items = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    pos = [[0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
    cands = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    res = NnAlgoSequentialRecommendationSasrec.forward(
        item_sequence_embeddings=items,
        position_embeddings=pos,
        candidate_item_embeddings=cands,
    )
    assert len(res["context_sequence"]) == 3
    assert len(res["candidate_scores"]) == 3
    assert 0 <= res["top_candidate_index"] < 3
    # Check lower triangular attention
    attn = res["causal_attention_matrix"]
    assert attn[0][1] == 0.0 and attn[0][2] == 0.0
    assert attn[1][2] == 0.0
    assert abs(sum(attn[2]) - 1.0) < 1e-6


def test_algo_193_mmoe():
    x = [0.5, -0.5]
    expert_w = [
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.1], [0.2, 0.3]],
    ]
    expert_b = [[0.0, 0.1], [0.1, 0.0]]
    gate_w = [
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.2, 0.1], [0.4, 0.3]],
    ]
    tower_w = [[0.5, 0.5], [-0.5, 0.5]]
    tower_b = [0.0, 0.0]
    res = NnAlgoMultiGateMixtureOfExpertsMmoe.forward(
        input_x=x,
        expert_weights=expert_w,
        expert_biases=expert_b,
        task_gate_weights=gate_w,
        task_tower_weights=tower_w,
        task_tower_biases=tower_b,
    )
    assert len(res["task_predictions"]) == 2
    assert len(res["task_gate_distributions"]) == 2
    assert all(0.0 <= p <= 1.0 for p in res["task_predictions"])
    assert abs(sum(res["task_gate_distributions"][0]) - 1.0) < 1e-6


def test_algo_194_din():
    cand = [1.0, 0.5]
    hist = [[0.8, 0.6], [0.1, 0.9], [0.9, 0.4]]
    # 4 * D = 8
    attn_w1 = [[0.1] * 8, [0.2] * 8]
    attn_b1 = [0.0, 0.0]
    attn_w2 = [0.5, 0.5]
    attn_b2 = 0.1
    res = NnAlgoDeepInterestNetworkAttention.forward(
        candidate_embedding=cand,
        history_embeddings=hist,
        attn_weights_w1=attn_w1,
        attn_bias_b1=attn_b1,
        attn_weights_w2=attn_w2,
        attn_bias_b2=attn_b2,
        normalize_softmax=True,
    )
    assert len(res["user_interest_vector"]) == 2
    assert len(res["attention_weights"]) == 3
    assert 0 <= res["dominant_history_index"] < 3
    assert abs(res["total_attention_mass"] - 1.0) < 1e-6


def test_algo_195_dlrm():
    dense = [0.5, 1.0, -0.5]
    sparse = [[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]
    bot_w = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # (2, 3)
    bot_b = [0.0, 0.1]  # D = 2
    # S = 3, interactions = 3 * 4 // 2 = 6. Top dim = 2 + 6 = 8
    top_w = [0.1] * 8
    top_b = 0.0
    res = NnAlgoDeepLearningRecommendationDlrm.forward(
        dense_features=dense,
        sparse_embeddings=sparse,
        bottom_mlp_weights=bot_w,
        bottom_mlp_bias=bot_b,
        top_mlp_weights=top_w,
        top_mlp_bias=top_b,
    )
    assert len(res["bottom_representation"]) == 2
    assert len(res["interaction_vector"]) == 6
    assert 0.0 <= res["predicted_probability"] <= 1.0
