import pytest
import math

from src.features.code_engine.algos.nn.advanced_reinforcement_learning.soft_actor_critic.impl import (
    NnAlgoSoftActorCritic,
)
from src.features.code_engine.algos.nn.advanced_reinforcement_learning.twin_delayed_ddpg_td3.impl import (
    NnAlgoTwinDelayedDdpgTd3,
)
from src.features.code_engine.algos.nn.advanced_reinforcement_learning.world_models_dreamer.impl import (
    NnAlgoWorldModelsDreamer,
)
from src.features.code_engine.algos.nn.advanced_reinforcement_learning.mcts_alphazero_muzero.impl import (
    NnAlgoMctsAlphazeroMuzero,
)
from src.features.code_engine.algos.nn.advanced_reinforcement_learning.offline_rl_cql_iql.impl import (
    NnAlgoOfflineRlCqlIql,
)
from src.features.code_engine.algos.nn.advanced_reinforcement_learning.imitation_learning_dagger.impl import (
    NnAlgoImitationLearningDagger,
)


def test_algo_201_soft_actor_critic():
    # Test certificate: If dones are all True, critic_target equals rewards
    res_done = NnAlgoSoftActorCritic.forward(
        current_q1_values=[1.0, 2.0],
        current_q2_values=[1.2, 1.8],
        next_q1_target_values=[3.0, 4.0],
        next_q2_target_values=[2.5, 4.5],
        rewards=[5.0, 10.0],
        next_log_probs=[-0.5, -0.8],
        dones=[True, True],
        policy_log_probs=[-0.3, -0.4],
        policy_q1_values=[2.0, 3.0],
        policy_q2_values=[1.8, 2.8],
        alpha=0.2,
        gamma=0.99,
    )
    assert res_done["critic_target"] == [5.0, 10.0]
    assert res_done["updated_alpha"] > 0.0

    # Test soft Bellman target for non-terminal
    # min(2.5, 3.0) - 0.2 * (-0.5) = 2.5 + 0.1 = 2.6 -> Target = 1.0 + 0.99 * 2.6 = 3.574
    res_non_done = NnAlgoSoftActorCritic.forward(
        current_q1_values=[1.0],
        current_q2_values=[1.0],
        next_q1_target_values=[3.0],
        next_q2_target_values=[2.5],
        rewards=[1.0],
        next_log_probs=[-0.5],
        dones=[False],
        policy_log_probs=[-0.2],
        policy_q1_values=[2.0],
        policy_q2_values=[2.2],
        alpha=0.2,
        gamma=0.99,
    )
    assert abs(res_non_done["critic_target"][0] - 3.574) < 1e-5
    assert res_non_done["critic_loss"] > 0.0


def test_algo_202_td3():
    # Test certificate: If current_step % policy_delay != 0, actor_updated is False and actor_loss is 0.0
    res_delay = NnAlgoTwinDelayedDdpgTd3.forward(
        current_q1_values=[2.0, 3.0],
        current_q2_values=[2.1, 2.9],
        target_q1_values=[4.0, 5.0],
        target_q2_values=[3.8, 5.2],
        rewards=[1.0, 0.5],
        dones=[False, False],
        actor_q1_values=[3.5, 4.0],
        target_action_noise=[0.8, -0.9],
        noise_clip=0.5,
        policy_delay=2,
        current_step=1,
    )
    assert res_delay["actor_updated"] is False
    assert res_delay["actor_loss"] == 0.0
    assert res_delay["effective_noise"] == [0.5, -0.5]

    # Delayed update active on step 2
    res_step2 = NnAlgoTwinDelayedDdpgTd3.forward(
        current_q1_values=[2.0, 3.0],
        current_q2_values=[2.1, 2.9],
        target_q1_values=[4.0, 5.0],
        target_q2_values=[3.8, 5.2],
        rewards=[1.0, 0.5],
        dones=[False, False],
        actor_q1_values=[3.5, 4.5],
        target_action_noise=[0.1, -0.1],
        noise_clip=0.5,
        policy_delay=2,
        current_step=2,
    )
    assert res_step2["actor_updated"] is True
    assert abs(res_step2["actor_loss"] - (-4.0)) < 1e-6


def test_algo_203_dreamer():
    # Test certificate: If posterior equals prior and predictions match targets, world_model_loss is 0.0
    res_cert = NnAlgoWorldModelsDreamer.forward(
        observations=[1.0, 2.0],
        reconstructed_observations=[1.0, 2.0],
        rewards=[0.5, 1.0],
        predicted_rewards=[0.5, 1.0],
        posterior_means=[0.0, 0.0],
        posterior_logvars=[0.0, 0.0],
        prior_means=[0.0, 0.0],
        prior_logvars=[0.0, 0.0],
        imagined_rewards=[1.0, 1.0],
        imagined_values=[2.0, 2.0, 2.0],
        kl_scale=1.0,
    )
    assert abs(res_cert["world_model_loss"]) < 1e-6
    assert abs(res_cert["kl_divergence"]) < 1e-6
    assert len(res_cert["lambda_returns"]) == 2


def test_algo_204_mcts_alphazero():
    # Test certificate: If all visit counts are equal and temperature is 1.0, target_policy is uniform 1/A
    res_uniform = NnAlgoMctsAlphazeroMuzero.forward(
        prior_probabilities=[0.25, 0.25, 0.25, 0.25],
        visit_counts=[10, 10, 10, 10],
        q_values=[1.0, 1.0, 1.0, 1.0],
        temperature=1.0,
    )
    for p in res_uniform["target_policy"]:
        assert abs(p - 0.25) < 1e-6

    # Test PUCT selection
    res_select = NnAlgoMctsAlphazeroMuzero.forward(
        prior_probabilities=[0.1, 0.8, 0.1],
        visit_counts=[5, 50, 2],
        q_values=[0.5, 0.9, 0.4],
    )
    assert res_select["selected_action"] in [0, 1, 2]
    assert len(res_select["puct_scores"]) == 3


def test_algo_205_offline_rl_cql_iql():
    # Test certificate: If dataset_q equals bellman_target and candidate contains only dataset_q, cql_penalty is 0.0
    res_cert = NnAlgoOfflineRlCqlIql.forward(
        dataset_q_values=[2.0, 3.0],
        candidate_q_values=[[2.0], [3.0]],
        bellman_targets=[2.0, 3.0],
        current_v_values=[2.0, 3.0],
        cql_alpha=1.0,
        iql_expectile=0.7,
    )
    assert abs(res_cert["bellman_loss"]) < 1e-6
    assert abs(res_cert["cql_penalty"]) < 1e-6
    assert abs(res_cert["cql_critic_loss"]) < 1e-6
    assert abs(res_cert["iql_value_loss"]) < 1e-6


def test_algo_206_imitation_learning_dagger():
    # Test certificate: If policy_predicted_actions equals expert_actions, iteration_imitation_loss is 0.0
    res_cert = NnAlgoImitationLearningDagger.forward(
        policy_predicted_actions=[1.0, 2.5, -0.5],
        expert_actions=[1.0, 2.5, -0.5],
        current_iteration=1,
        initial_beta=0.5,
        history_dataset_size=100,
    )
    assert abs(res_cert["iteration_imitation_loss"]) < 1e-6
    assert res_cert["aggregated_dataset_size"] == 103
    assert abs(res_cert["beta_mixing_probability"] - 1.0) < 1e-6

    # Test iteration 2 beta decay: beta_2 = 0.5^(2-1) = 0.5
    res_it2 = NnAlgoImitationLearningDagger.forward(
        policy_predicted_actions=[1.0, 2.0],
        expert_actions=[1.2, 1.9],
        current_iteration=2,
        initial_beta=0.5,
        history_dataset_size=103,
    )
    assert abs(res_it2["beta_mixing_probability"] - 0.5) < 1e-6
    assert res_it2["iteration_imitation_loss"] > 0.0
    assert res_it2["dagger_error_bound"] < res_it2["bc_error_bound"]
