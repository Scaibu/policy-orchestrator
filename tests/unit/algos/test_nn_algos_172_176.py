import pytest
import math

from src.features.code_engine.algos.nn.self_supervised_learning.non_contrastive_self_supervision.impl import NnAlgoNonContrastiveSelfSupervision
from src.features.code_engine.algos.nn.self_supervised_learning.redundancy_reduction_ssl.impl import NnAlgoRedundancyReductionSsl
from src.features.code_engine.algos.nn.self_supervised_learning.masked_autoencoders_vision.impl import NnAlgoMaskedAutoencodersVision
from src.features.code_engine.algos.nn.self_supervised_learning.self_distillation_dino.impl import NnAlgoSelfDistillationDino
from src.features.code_engine.algos.nn.self_supervised_learning.joint_embedding_predictive_jepa.impl import NnAlgoJointEmbeddingPredictiveJepa


def test_algo_172_non_contrastive_self_supervision():
    p1 = [1.0, 0.0, 0.5]
    p2 = [0.0, 1.0, 0.5]
    z1 = [0.9, 0.1, 0.4]
    z2 = [0.1, 0.8, 0.6]

    res = NnAlgoNonContrastiveSelfSupervision.forward(p1, p2, z1, z2)
    assert "symmetric_loss" in res
    assert "cosine_similarity_12" in res
    assert "cosine_similarity_21" in res
    assert "representation_std" in res
    assert res["symmetric_loss"] >= 0.0
    assert -1.0 <= res["cosine_similarity_12"] <= 1.0
    assert -1.0 <= res["cosine_similarity_21"] <= 1.0


def test_algo_173_redundancy_reduction_ssl():
    z1 = [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]
    z2 = [[1.1, 1.9], [2.8, 1.2], [2.1, 3.9]]

    res = NnAlgoRedundancyReductionSsl.forward(z1, z2, lambda_off_diagonal=0.005)
    assert res["barlow_loss"] >= 0.0
    assert res["on_diagonal_invariance_loss"] >= 0.0
    assert res["off_diagonal_redundancy_loss"] >= 0.0
    assert abs(res["barlow_loss"] - (res["on_diagonal_invariance_loss"] + 0.005 * res["off_diagonal_redundancy_loss"])) < 1e-6


def test_algo_174_masked_autoencoders_vision():
    orig = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    recon = [[1.1, 1.9, 3.0], [0.0, 0.0, 0.0], [7.2, 7.9, 9.1], [0.0, 0.0, 0.0]]
    mask = [0, 1, 0, 1]

    res = NnAlgoMaskedAutoencodersVision.forward(orig, recon, mask, normalize_pixels=False)
    assert res["num_masked_patches"] == 2
    assert res["num_visible_patches"] == 2
    assert res["masking_ratio"] == 0.5
    assert res["masked_patch_loss"] >= 0.0
    assert res["overall_reconstruction_loss"] >= 0.0


def test_algo_175_self_distillation_dino():
    s_logits = [2.0, 1.0, 0.1, -1.0]
    t_logits = [3.0, 1.5, 0.0, -0.5]
    center = [0.0, 0.0, 0.0, 0.0]

    res = NnAlgoSelfDistillationDino.forward(
        s_logits, t_logits, center, tau_student=0.1, tau_teacher=0.04, center_momentum=0.9
    )
    assert res["cross_entropy_loss"] >= 0.0
    assert len(res["updated_center_vector"]) == 4
    assert res["teacher_entropy"] >= 0.0
    assert res["student_entropy"] >= 0.0
    assert 0.0 <= res["max_teacher_prob"] <= 1.0


def test_algo_176_joint_embedding_predictive_jepa():
    pred = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    target = [[1.1, 1.9, 3.0], [4.1, 4.9, 6.2]]

    res_l1 = NnAlgoJointEmbeddingPredictiveJepa.forward(pred, target, loss_type="l1")
    assert res_l1["jepa_prediction_loss"] == res_l1["l1_discrepancy"]
    assert res_l1["l1_discrepancy"] >= 0.0
    assert res_l1["l2_discrepancy"] >= 0.0
    assert -1.0 <= res_l1["mean_cosine_similarity"] <= 1.0

    res_l2 = NnAlgoJointEmbeddingPredictiveJepa.forward(pred, target, loss_type="l2")
    assert res_l2["jepa_prediction_loss"] == res_l2["l2_discrepancy"]
