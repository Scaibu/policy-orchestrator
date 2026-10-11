import pytest
import math

from src.features.code_engine.algos.nn.llm_alignment_post_training.curriculum_learning.impl import (
    NnAlgoCurriculumLearning,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.supervised_fine_tuning.impl import (
    NnAlgoSupervisedFineTuning,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.reward_modeling_bradley_terry.impl import (
    NnAlgoRewardModelingBradleyTerry,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.rlhf_ppo_kl_penalty.impl import (
    NnAlgoRlhfPpoKlPenalty,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.direct_preference_optimization_dpo.impl import (
    NnAlgoDirectPreferenceOptimizationDpo,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.dpo_variants_simpo_kto.impl import (
    NnAlgoDpoVariantsSimpoKto,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.group_relative_policy_optimization_grpo.impl import (
    NnAlgoGroupRelativePolicyOptimizationGrpo,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.rl_verifiable_rewards_rlvr.impl import (
    NnAlgoRlVerifiableRewardsRlvr,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.rlaif_constitution_critique.impl import (
    NnAlgoRlaifConstitutionCritique,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.rejection_sampling_best_of_n.impl import (
    NnAlgoRejectionSamplingBestOfN,
)
from src.features.code_engine.algos.nn.llm_alignment_post_training.process_reward_models_prm.impl import (
    NnAlgoProcessRewardModelsPrm,
)


def test_algo_207_curriculum_learning():
    diffs = [5.0, 1.0, 3.0, 2.0, 4.0]
    # Step 0: fraction = initial_fraction (0.2) -> 1 sample (easiest: index 1 with diff 1.0)
    res_start = NnAlgoCurriculumLearning.forward(
        sample_difficulties=diffs,
        current_step=0,
        total_pacing_steps=100,
        initial_fraction=0.2,
    )
    assert res_start["eligible_indices"] == [1]
    assert abs(res_start["current_difficulty_threshold"] - 1.0) < 1e-6
    assert abs(res_start["retention_fraction"] - 0.2) < 1e-6

    # Step >= total_steps: retention is 1.0
    res_end = NnAlgoCurriculumLearning.forward(
        sample_difficulties=diffs,
        current_step=100,
        total_pacing_steps=100,
        initial_fraction=0.2,
    )
    assert res_end["retention_fraction"] == 1.0
    assert len(res_end["eligible_indices"]) == 5


def test_algo_208_supervised_fine_tuning():
    targets = [0, 1, 2]
    # Mock logits where target gets 10.0 and others -10.0
    logits = [
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
    ]
    mask = [0, 1, 1]  # first token prompt, last 2 assistant
    res = NnAlgoSupervisedFineTuning.forward(
        target_tokens=targets,
        predicted_logits=logits,
        loss_mask=mask,
    )
    assert res["active_token_count"] == 2
    assert res["token_accuracy"] == 1.0
    assert res["masked_sft_loss"] < 1e-3
    assert abs(res["completion_perplexity"] - 1.0) < 1e-3


def test_algo_209_reward_modeling_bradley_terry():
    chosen = [5.0, 10.0]
    rejected = [1.0, 2.0]
    lengths_w = [100, 150]
    lengths_l = [80, 140]
    res = NnAlgoRewardModelingBradleyTerry.forward(
        chosen_rewards=chosen,
        rejected_rewards=rejected,
        chosen_lengths=lengths_w,
        rejected_lengths=lengths_l,
    )
    assert res["pairwise_accuracy"] == 1.0
    assert res["bradley_terry_loss"] < 0.05
    assert res["mean_reward_margin"] == 6.0


def test_algo_210_rlhf_ppo_kl_penalty():
    # If policy matches ref identically:
    log_probs = [-0.5, -0.2]
    old_lps = [-0.5, -0.2]
    ref_lps = [-0.5, -0.2]
    values = [1.0, 2.0]
    res = NnAlgoRlhfPpoKlPenalty.forward(
        policy_log_probs=log_probs,
        old_log_probs=old_lps,
        ref_log_probs=ref_lps,
        value_estimates=values,
        sequence_reward=2.0,
        kl_penalty_coef=0.1,
    )
    assert abs(res["total_kl_divergence"]) < 1e-6
    assert sum(res["penalized_rewards"]) == 2.0


def test_algo_211_direct_preference_optimization_dpo():
    # Certificate: If policy == ref on all pairs, implicit rewards are 0.0 and loss is ln(2)
    lps = [-1.0, -2.0]
    res_cert = NnAlgoDirectPreferenceOptimizationDpo.forward(
        policy_chosen_log_probs=lps,
        policy_rejected_log_probs=lps,
        ref_chosen_log_probs=lps,
        ref_rejected_log_probs=lps,
        beta=0.1,
    )
    assert res_cert["implicit_chosen_rewards"] == [0.0, 0.0]
    assert res_cert["implicit_rejected_rewards"] == [0.0, 0.0]
    assert abs(res_cert["dpo_loss"] - math.log(2.0)) < 1e-6


def test_algo_212_dpo_variants():
    # SimPO test: length-normalized margin
    c_lps = [-10.0]  # length 10 -> -1.0 per token
    r_lps = [-30.0]  # length 10 -> -3.0 per token
    res_simpo = NnAlgoDpoVariantsSimpoKto.forward(
        chosen_log_probs=c_lps,
        chosen_lengths=[10],
        variant_type="simpo",
        rejected_log_probs=r_lps,
        rejected_lengths=[10],
        beta=1.0,
        target_margin=0.5,
    )
    # margin = 1.0 * (-1.0) - 1.0 * (-3.0) = 2.0 > 0.5
    assert abs(res_simpo["margin_scores"][0] - 2.0) < 1e-6
    assert res_simpo["variant_accuracy"] == 1.0


def test_algo_213_grpo():
    rewards = [2.0, 2.0, 2.0]
    lps = [[-0.5, -0.5], [-0.5, -0.5], [-0.5, -0.5]]
    # Certificate: if all rewards are identical, advantages are 0.0
    res = NnAlgoGroupRelativePolicyOptimizationGrpo.forward(
        group_rewards=rewards,
        group_policy_log_probs=lps,
        group_old_log_probs=lps,
        group_ref_log_probs=lps,
    )
    assert res["group_advantages"] == [0.0, 0.0, 0.0]
    assert res["std_group_reward"] == 0.0


def test_algo_214_rlvr():
    tests = [[True, True], [True, False]]
    fmts = [True, False]
    lens = [500, 2000]
    res = NnAlgoRlVerifiableRewardsRlvr.forward(
        test_results=tests,
        format_valid=fmts,
        response_lengths=lens,
        max_target_length=1000,
        weight_pass=1.0,
        weight_format=0.2,
    )
    assert res["all_tests_passed"] == [True, False]
    assert res["pass_at_1"] == 0.5
    # First candidate: 1.0 + 0.2 - 0 = 1.2
    assert abs(res["total_rewards"][0] - 1.2) < 1e-6


def test_algo_215_rlaif_constitution():
    # Certificate: if all critique scores are 1.0, composite is sum(weights)
    critiques = [[1.0, 1.0], [0.5, 0.9]]
    weights = [2.0, 3.0]
    res = NnAlgoRlaifConstitutionCritique.forward(
        critique_scores=critiques,
        principle_weights=weights,
        satisfaction_threshold=0.8,
    )
    assert abs(res["composite_constitutional_scores"][0] - 5.0) < 1e-6
    assert res["all_principles_satisfied"] == [True, False]
    assert res["worst_violated_principles"][1] == 0


def test_algo_216_best_of_n():
    scores = [0.2, 0.9, 0.85]
    tokens = [[1, 2, 3], [4, 5, 6], [4, 5, 7]]  # candidate 1 and 2 share tokens
    res = NnAlgoRejectionSamplingBestOfN.forward(
        candidate_scores=scores,
        candidate_tokens=tokens,
        rejection_threshold=0.5,
        top_k=2,
        dedup_similarity_threshold=0.5,  # [4,5,6] & [4,5,7] overlap 2/4 = 0.5
    )
    assert res["best_candidate_index"] == 1
    assert res["max_score"] == 0.9
    assert res["accepted_candidate_indices"] == [1]
    assert res["diversity_filtered_count"] == 1


def test_algo_217_prm():
    preds = [0.95, 0.9, 0.2, 0.8]
    gt = [1, 1, 0, 1]
    res = NnAlgoProcessRewardModelsPrm.forward(
        step_predictions=preds,
        step_ground_truth=gt,
        error_threshold=0.5,
        aggregation_method="min",
    )
    assert res["first_error_step"] == 2
    assert abs(res["composite_solution_score"] - 0.2) < 1e-6
    assert res["step_accuracy"] == 1.0
