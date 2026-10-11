import pytest
from src.features.code_engine.algos.nn.distillation_continual_learning.knowledge_distillation_soft_targets.impl import (
    NnAlgoKnowledgeDistillationSoftTargets,
)
from src.features.code_engine.algos.nn.distillation_continual_learning.sequence_on_policy_distillation.impl import (
    NnAlgoSequenceOnPolicyDistillation,
)
from src.features.code_engine.algos.nn.distillation_continual_learning.elastic_weight_consolidation_ewc.impl import (
    NnAlgoElasticWeightConsolidationEwc,
)
from src.features.code_engine.algos.nn.distillation_continual_learning.experience_replay_continual_learning.impl import (
    NnAlgoExperienceReplayContinualLearning,
)


def test_algo_225_knowledge_distillation():
    student_logits = [2.0, 0.5, -1.0]
    teacher_logits = [3.0, 1.0, -0.5]
    true_label = 0

    res = NnAlgoKnowledgeDistillationSoftTargets.forward(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        true_label=true_label,
        temperature=2.0,
        alpha=0.7,
    )
    assert res["total_loss"] >= 0.0
    assert res["distillation_kl_loss"] >= 0.0
    assert res["hard_ce_loss"] >= 0.0
    assert len(res["student_soft_probabilities"]) == 3
    assert abs(sum(res["student_soft_probabilities"]) - 1.0) < 1e-5
    assert abs(sum(res["teacher_soft_probabilities"]) - 1.0) < 1e-5
    assert res["student_pred_class"] == 0
    assert res["teacher_pred_class"] == 0


def test_algo_226_sequence_on_policy_distillation():
    student_logits = [
        [1.0, 0.2, 0.1],
        [0.5, 2.0, 0.3],
    ]
    teacher_logits = [
        [1.5, 0.1, 0.0],
        [0.2, 2.5, 0.1],
    ]
    token_ids = [0, 1]

    # Test forward_kl
    res_fwd = NnAlgoSequenceOnPolicyDistillation.forward(
        student_sequence_logits=student_logits,
        teacher_sequence_logits=teacher_logits,
        token_ids=token_ids,
        distillation_mode="forward_kl",
    )
    assert len(res_fwd["token_losses"]) == 2
    assert res_fwd["mean_sequence_loss"] >= 0.0
    assert res_fwd["effective_sequence_length"] == 2
    assert len(res_fwd["per_token_student_entropies"]) == 2

    # Test reverse_kl
    res_rev = NnAlgoSequenceOnPolicyDistillation.forward(
        student_sequence_logits=student_logits,
        teacher_sequence_logits=teacher_logits,
        token_ids=token_ids,
        distillation_mode="reverse_kl",
    )
    assert res_rev["mean_sequence_loss"] >= 0.0

    # Test sequence_sft
    res_sft = NnAlgoSequenceOnPolicyDistillation.forward(
        student_sequence_logits=student_logits,
        teacher_sequence_logits=teacher_logits,
        token_ids=token_ids,
        distillation_mode="sequence_sft",
    )
    assert res_sft["mean_sequence_loss"] >= 0.0


def test_algo_227_ewc():
    current_w = [1.2, 0.8, -0.4, 0.1]
    optimal_w = [1.0, 1.0, -0.5, 0.0]
    fisher = [10.0, 5.0, 2.0, 1.0]

    res = NnAlgoElasticWeightConsolidationEwc.forward(
        current_weights=current_w,
        old_optimal_weights=optimal_w,
        fisher_information_diagonal=fisher,
        ewc_lambda=50.0,
        task_b_loss=0.35,
    )
    assert res["ewc_penalty"] > 0.0
    assert res["total_loss"] > res["task_b_loss"]
    assert len(res["regularized_gradients"]) == 4
    assert res["max_weight_drift"] > 0.0
    assert res["active_parameter_count"] == 4

    # If weights are unchanged, penalty must be zero
    res_zero = NnAlgoElasticWeightConsolidationEwc.forward(
        current_weights=optimal_w,
        old_optimal_weights=optimal_w,
        fisher_information_diagonal=fisher,
        ewc_lambda=50.0,
        task_b_loss=0.35,
    )
    assert abs(res_zero["ewc_penalty"]) < 1e-9
    assert abs(res_zero["total_loss"] - 0.35) < 1e-9


def test_algo_228_experience_replay():
    new_batch = [{"id": 1, "text": "new_1"}, {"id": 2, "text": "new_2"}]
    replay_buf = [{"id": 100, "text": "old_1"}, {"id": 101, "text": "old_2"}, {"id": 102, "text": "old_3"}]

    res = NnAlgoExperienceReplayContinualLearning.forward(
        new_task_batch=new_batch,
        replay_buffer=replay_buf,
        replay_ratio=0.5,
        max_buffer_capacity=5,
        sampling_strategy="fifo",
    )
    assert len(res["blended_batch"]) == res["replay_sample_count"] + res["new_sample_count"]
    assert res["replay_sample_count"] > 0
    assert res["new_sample_count"] == 2
    assert len(res["updated_buffer"]) <= 5
    assert res["buffer_utilization"] <= 1.0
