import pytest
import math

from src.features.code_engine.algos.nn.peft_model_merging.lora_low_rank_adaptation.impl import (
    NnAlgoLoraLowRankAdaptation,
)
from src.features.code_engine.algos.nn.peft_model_merging.qlora_nf4_quantization.impl import (
    NnAlgoQloraNf4Quantization,
)
from src.features.code_engine.algos.nn.peft_model_merging.dora_weight_decomposed_lora.impl import (
    NnAlgoDoraWeightDecomposedLora,
)
from src.features.code_engine.algos.nn.peft_model_merging.bottleneck_adapter_modules.impl import (
    NnAlgoBottleneckAdapterModules,
)
from src.features.code_engine.algos.nn.peft_model_merging.prefix_prompt_tuning.impl import (
    NnAlgoPrefixPromptTuning,
)
from src.features.code_engine.algos.nn.peft_model_merging.ia3_bitfit_minimal_tuning.impl import (
    NnAlgoIa3BitfitMinimalTuning,
)
from src.features.code_engine.algos.nn.peft_model_merging.model_merging_ties_dare.impl import (
    NnAlgoModelMergingTiesDare,
)


def test_algo_218_lora():
    # If B is 0, output equals base W_0 * x
    x = [1.0, 2.0]
    w0 = [[2.0, 0.0], [0.0, 3.0]]
    a = [[1.0, 1.0]]
    b = [[0.0], [0.0]]
    res = NnAlgoLoraLowRankAdaptation.forward(
        input_vector=x,
        base_weight=w0,
        lora_a=a,
        lora_b=b,
        lora_alpha=1.0,
        lora_rank=1,
    )
    assert res["output_vector"] == [2.0, 6.0]
    assert res["base_output_vector"] == [2.0, 6.0]
    assert res["lora_delta_vector"] == [0.0, 0.0]


def test_algo_219_qlora():
    # Certificate: NF4 indices bounded in [0, 15]
    raw_w = [0.1, -0.5, 0.8, -0.2]
    x = [1.0, 1.0]
    a = [[0.1, 0.2]]
    b = [[0.0], [0.0]]
    res = NnAlgoQloraNf4Quantization.forward(
        raw_weights=raw_w,
        input_vector=x,
        lora_a=a,
        lora_b=b,
        block_size=2,
        lora_rank=1,
    )
    assert all(0 <= idx <= 15 for idx in res["quantized_nf4_indices"])
    assert res["quantization_mse_error"] >= 0.0
    assert len(res["dequantized_weights"]) == 4


def test_algo_220_dora():
    x = [1.0, 0.0]
    w0 = [[2.0, 0.0], [0.0, 2.0]]
    a = [[0.0, 0.0]]
    b = [[0.0], [0.0]]
    # Magnitude vector matching original row norms: [2.0, 2.0]
    res = NnAlgoDoraWeightDecomposedLora.forward(
        input_vector=x,
        base_weight=w0,
        lora_a=a,
        lora_b=b,
        magnitude_vector=[2.0, 2.0],
        lora_rank=1,
    )
    assert abs(res["output_vector"][0] - 2.0) < 1e-5
    assert abs(res["output_vector"][1] - 0.0) < 1e-5


def test_algo_221_bottleneck_adapters():
    # If W_up is 0, output matches input exactly (residual)
    x = [1.0, 2.0]
    w_down = [[0.5, 0.5]]
    b_down = [0.0]
    w_up = [[0.0], [0.0]]
    b_up = [0.0, 0.0]
    res = NnAlgoBottleneckAdapterModules.forward(
        input_vector=x,
        w_down=w_down,
        b_down=b_down,
        w_up=w_up,
        b_up=b_up,
    )
    assert res["output_vector"] == [1.0, 2.0]
    assert res["adapter_residual_delta"] == [0.0, 0.0]


def test_algo_222_prefix_prompt_tuning():
    q = [1.0, 0.0]
    k = [[1.0, 0.0]]
    v = [[2.0, 3.0]]
    pk = [[0.0, 1.0]]
    pv = [[4.0, 5.0]]
    res = NnAlgoPrefixPromptTuning.forward(
        query_vector=q,
        key_vectors=k,
        value_vectors=v,
        prefix_keys=pk,
        prefix_values=pv,
    )
    assert len(res["attention_output"]) == 2
    assert abs(sum(res["attention_weights"]) - 1.0) < 1e-5
    assert 0.0 <= res["prefix_attention_mass"] <= 1.0


def test_algo_223_ia3_bitfit():
    # Certificate: If scale is 1.0 and bias is 0.0, output matches activation
    x = [2.0, -3.0]
    scale = [1.0, 1.0]
    bias = [0.0, 0.0]
    res = NnAlgoIa3BitfitMinimalTuning.forward(
        activation_vector=x,
        ia3_scale_vector=scale,
        trainable_bias=bias,
        mode="combined",
    )
    assert res["modulated_output"] == [2.0, -3.0]
    assert res["parameter_count"] == 4


def test_algo_224_model_merging():
    base = [1.0, 2.0]
    task1 = [1.5, 2.5]
    task2 = [1.5, 2.5]
    res = NnAlgoModelMergingTiesDare.forward(
        base_weights=base,
        task_weights=[task1, task2],
        merge_method="ties",
        top_k_fraction=1.0,
    )
    assert res["merged_weights"] == [1.5, 2.5]
    assert res["effective_task_vector"] == [0.5, 0.5]
    assert res["sign_consensus_ratio"] == 1.0
