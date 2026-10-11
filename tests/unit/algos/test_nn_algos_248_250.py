import pytest
from src.features.code_engine.algos.nn.pruning_sparsity.magnitude_pruning_lottery_ticket.impl import (
    NnAlgoMagnitudePruningLotteryTicket,
)
from src.features.code_engine.algos.nn.pruning_sparsity.structured_oneshot_pruning_llms.impl import (
    NnAlgoStructuredOneshotPruningLlms,
)
from src.features.code_engine.algos.nn.pruning_sparsity.semistructured_2to4_sparsity.impl import (
    NnAlgoSemistructured2to4Sparsity,
)


def test_algo_248_magnitude_pruning_lottery_ticket():
    current_w = [0.1, -0.9, 0.4, -0.05, 0.8]
    initial_w = [0.05, -0.5, 0.2, -0.01, 0.4]

    # Prune 40% (2 smallest absolute weights: -0.05 and 0.1)
    res = NnAlgoMagnitudePruningLotteryTicket.forward(
        current_weights=current_w,
        initial_weights=initial_w,
        target_sparsity=0.4,
        rewind_weights=True,
    )
    assert res["binary_mask"][0] == 0  # 0.1 pruned
    assert res["binary_mask"][3] == 0  # -0.05 pruned
    assert res["binary_mask"][1] == 1  # -0.9 kept
    assert res["binary_mask"][4] == 1  # 0.8 kept
    assert res["pruned_weights"][1] == initial_w[1]  # rewound to initial
    assert res["surviving_parameter_count"] == 3


def test_algo_249_structured_oneshot_pruning_wanda():
    weights = [
        [1.0, 2.0, 0.5, 4.0],
        [0.2, 0.1, 3.0, 0.8],
    ]
    act_norms = [1.0, 0.5, 2.0, 0.1]
    # Wanda score = |W| * ||X||:
    # Row 0: [1.0*1.0, 2.0*0.5, 0.5*2.0, 4.0*0.1] = [1.0, 1.0, 1.0, 0.4] -> min is index 3
    # Row 1: [0.2*1.0, 0.1*0.5, 3.0*2.0, 0.8*0.1] = [0.2, 0.05, 6.0, 0.08] -> min is index 1

    res = NnAlgoStructuredOneshotPruningLlms.forward(
        weight_matrix=weights,
        activation_norms=act_norms,
        pruning_ratio=0.25,
        mode="wanda",
    )
    assert res["pruned_matrix"][0][3] == 0.0
    assert res["pruned_matrix"][1][1] == 0.0
    assert res["zeroed_parameter_count"] == 2
    assert res["achieved_sparsity"] == 0.25


def test_algo_250_semistructured_2to4_sparsity():
    dense_vec = [1.0, -5.0, 0.2, 3.0, -0.1, 4.0, -8.0, 0.5]
    # Block 0: [1.0, -5.0, 0.2, 3.0] -> magnitudes [1.0, 5.0, 0.2, 3.0]. Top 2: -5.0 (idx 1), 3.0 (idx 3)
    # Block 1: [-0.1, 4.0, -8.0, 0.5] -> magnitudes [0.1, 4.0, 8.0, 0.5]. Top 2: 4.0 (idx 5), -8.0 (idx 6)

    res = NnAlgoSemistructured2to4Sparsity.forward(
        dense_vector=dense_vec,
    )
    assert res["sparse_vector"][0] == 0.0
    assert res["sparse_vector"][1] == -5.0
    assert res["sparse_vector"][2] == 0.0
    assert res["sparse_vector"][3] == 3.0
    assert len(res["compressed_values"]) == 4
    assert res["compression_ratio"] == 0.5
    assert res["block_count"] == 2
