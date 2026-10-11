import pytest
import math

from src.features.code_engine.algos.nn.reinforcement_learning.deep_q_networks_dqn.impl import (
    NnAlgoDeepQNetworksDqn,
)
from src.features.code_engine.algos.nn.reinforcement_learning.rainbow_dqn_extensions.impl import (
    NnAlgoRainbowDqnExtensions,
)
from src.features.code_engine.algos.nn.reinforcement_learning.policy_gradients_reinforce.impl import (
    NnAlgoPolicyGradientsReinforce,
)
from src.features.code_engine.algos.nn.reinforcement_learning.actor_critic_gae.impl import (
    NnAlgoActorCriticGae,
)
from src.features.code_engine.algos.nn.reinforcement_learning.proximal_policy_optimization_ppo.impl import (
    NnAlgoProximalPolicyOptimizationPpo,
)


def test_algo_196_dqn():
    q_vals = [1.5, 2.0, 0.5]
    next_q = [2.5, 1.0, 3.0]
    # Non-terminal
    res = NnAlgoDeepQNetworksDqn.forward(
        current_q_values=q_vals,
        action_taken=1,
        reward=1.0,
        next_state_target_q_values=next_q,
        gamma_discount=0.9,
        is_terminal=False,
    )
    # Target = 1.0 + 0.9 * 3.0 = 3.7
    assert abs(res["td_target"] - 3.7) < 1e-6
    # Error = 3.7 - 2.0 = 1.7
    assert abs(res["td_error"] - 1.7) < 1e-6
    assert res["greedy_action"] == 1
    assert res["huber_loss"] > 0.0

    # Terminal
    res_term = NnAlgoDeepQNetworksDqn.forward(
        current_q_values=q_vals,
        action_taken=1,
        reward=5.0,
        next_state_target_q_values=next_q,
        gamma_discount=0.9,
        is_terminal=True,
    )
    assert abs(res_term["td_target"] - 5.0) < 1e-6


def test_algo_197_rainbow_dqn():
    v = 2.0
    adv = [1.0, -1.0, 0.0]  # mean = 0.0
    next_online = [3.0, 1.0, 2.0]  # argmax = 0
    next_target = [2.5, 4.0, 1.0]  # target evaluates action 0 -> 2.5
    res = NnAlgoRainbowDqnExtensions.forward(
        state_value_v=v,
        advantages_a=adv,
        action_taken=0,
        reward=1.0,
        next_online_q_values=next_online,
        next_target_q_values=next_target,
        gamma_discount=0.9,
    )
    # Synthesized Q = 2.0 + (adv - 0) = [3.0, 1.0, 2.0]
    assert res["synthesized_q_values"] == [3.0, 1.0, 2.0]
    # Double target = 1.0 + 0.9 * 2.5 = 3.25
    assert abs(res["double_td_target"] - 3.25) < 1e-6
    # TD error = 3.25 - 3.0 = 0.25
    assert abs(res["double_td_error"] - 0.25) < 1e-6
    assert res["per_transition_priority"] > 0.0
    assert res["greedy_action"] == 0


def test_algo_198_reinforce():
    probs = [0.8, 0.6, 0.7]
    rewards = [1.0, 2.0, 3.0]
    baselines = [0.5, 1.0, 1.5]
    res = NnAlgoPolicyGradientsReinforce.forward(
        action_probabilities=probs,
        step_rewards=rewards,
        baseline_values=baselines,
        gamma_discount=1.0,
        normalize_advantages=False,
    )
    # G = [6.0, 5.0, 3.0]
    assert res["discounted_returns"] == [6.0, 5.0, 3.0]
    # A = G - b = [5.5, 4.0, 1.5]
    assert res["advantages"] == [5.5, 4.0, 1.5]
    assert res["trajectory_return"] == 6.0


def test_algo_199_actor_critic_gae():
    probs = [0.5, 0.5]
    rewards = [1.0, 2.0]
    v = [1.0, 1.5]
    next_v = [1.5, 0.0]
    dones = [False, True]
    res = NnAlgoActorCriticGae.forward(
        action_probabilities=probs,
        step_rewards=rewards,
        state_values=v,
        next_state_values=next_v,
        dones=dones,
        gamma_discount=0.9,
        gae_lambda=0.9,
    )
    assert len(res["td_residuals"]) == 2
    assert len(res["gae_advantages"]) == 2
    assert len(res["value_targets"]) == 2
    assert res["critic_loss"] >= 0.0


def test_algo_200_ppo():
    cur_lp = [-0.693, -0.693, -0.693]  # log(0.5)
    old_lp = [-0.693, -0.693, -0.693]
    adv = [1.0, -1.0, 0.5]
    res = NnAlgoProximalPolicyOptimizationPpo.forward(
        current_log_probs=cur_lp,
        old_log_probs=old_lp,
        advantages=adv,
        clip_epsilon=0.2,
    )
    # Ratios must be exactly 1.0
    for r in res["probability_ratios"]:
        assert abs(r - 1.0) < 1e-6
    assert res["clip_fraction"] == 0.0
    assert abs(res["approximate_kl"] - 0.0) < 1e-6
