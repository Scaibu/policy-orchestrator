from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoActorCriticGae:
    """
    ---
    contract:
      algo_id: ALGO-NN-199
      name: NnAlgoActorCriticGae
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - actor_critic
        - gae
        - generalized_advantage_estimation
        - a2c
      inputs:
        type: object
        required:
          - action_probabilities
          - step_rewards
          - state_values
          - next_state_values
          - dones
        properties:
          action_probabilities:
            type: array
            items:
              type: number
            description: Selected action probabilities pi(a_t | s_t) of length T.
          step_rewards:
            type: array
            items:
              type: number
            description: Step rewards r_t of length T.
          state_values:
            type: array
            items:
              type: number
            description: Critic state-value estimations V(s_t) of length T.
          next_state_values:
            type: array
            items:
              type: number
            description: Critic next-state value estimations V(s_{t+1}) of length T.
          dones:
            type: array
            items:
              type: boolean
            description: Terminal state indicators of length T.
          gamma_discount:
            type: number
            default: 0.99
            minimum: 0.0
            maximum: 1.0
            description: Temporal discount factor gamma.
          gae_lambda:
            type: number
            default: 0.95
            minimum: 0.0
            maximum: 1.0
            description: GAE bias-variance exponential trade-off parameter lambda.
      outputs:
        type: object
        required:
          - td_residuals
          - gae_advantages
          - value_targets
          - actor_loss
          - critic_loss
        properties:
          td_residuals:
            type: array
            items:
              type: number
            description: 1-step TD residuals delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t).
          gae_advantages:
            type: array
            items:
              type: number
            description: Exponentially weighted GAE advantage estimates A_t^GAE.
          value_targets:
            type: array
            items:
              type: number
            description: Fitted value regression targets R_t = A_t^GAE + V(s_t).
          actor_loss:
            type: number
            description: Policy gradient surrogate loss - (1/T) sum log(pi) * A.
          critic_loss:
            type: number
            description: Value estimation mean squared error (1/T) sum (V(s_t) - R_t)^2.
      complexity:
        time: O(T)
        space: O(T)
      preconditions:
        - len(input.action_probabilities) > 0
        - len(input.step_rewards) == len(input.action_probabilities)
        - len(input.state_values) == len(input.action_probabilities)
        - len(input.next_state_values) == len(input.action_probabilities)
        - len(input.dones) == len(input.action_probabilities)
        - 0.0 <= input.gamma_discount <= 1.0
        - 0.0 <= input.gae_lambda <= 1.0
      postconditions:
        - len(output.td_residuals) == len(input.step_rewards)
        - len(output.gae_advantages) == len(input.step_rewards)
        - len(output.value_targets) == len(input.step_rewards)
        - output.critic_loss >= 0.0
      certificate: "GAE advantage at t=T-1 exactly matches 1-step TD residual delta_{T-1}"
      compatible_adapters:
        - ADAPTER-A2C-GAE-STEPPER
        - ADAPTER-PPO-ROLLOUT-BUFFER
      related_algos:
        - ALGO-NN-198
        - ALGO-NN-200
      references:
        - "https://arxiv.org/abs/1506.02438"
    ---
    """

    @classmethod
    def forward(
        cls,
        action_probabilities: List[float],
        step_rewards: List[float],
        state_values: List[float],
        next_state_values: List[float],
        dones: List[bool],
        gamma_discount: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Dict[str, Any]:
        T = len(step_rewards)
        if T == 0:
            raise ValueError("Precondition failed: step_rewards cannot be empty")
        if (
            len(action_probabilities) != T
            or len(state_values) != T
            or len(next_state_values) != T
            or len(dones) != T
        ):
            raise ValueError("Precondition failed: trajectory sequence dimensions mismatch")
        if not (0.0 <= gamma_discount <= 1.0) or not (0.0 <= gae_lambda <= 1.0):
            raise ValueError("Precondition failed: gamma_discount or gae_lambda out of bounds")

        eps = 1e-8

        # 1. Compute 1-step temporal difference residuals:
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        deltas: List[float] = []
        for t in range(T):
            next_v = 0.0 if dones[t] else next_state_values[t]
            delta = step_rewards[t] + gamma_discount * next_v - state_values[t]
            deltas.append(delta)

        # 2. Backward accumulation of Generalized Advantage Estimation:
        # A_t^GAE = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}^GAE
        gae_advantages = [0.0] * T
        running_adv = 0.0
        decay = gamma_discount * gae_lambda

        for t in reversed(range(T)):
            not_done = 0.0 if dones[t] else 1.0
            running_adv = deltas[t] + decay * not_done * running_adv
            gae_advantages[t] = running_adv

        # 3. Value targets: R_t = A_t^GAE + V(s_t)
        value_targets = [gae_advantages[t] + state_values[t] for t in range(T)]

        # 4. Losses:
        # Actor: - (1/T) * sum ln(pi_t) * A_t
        # Critic: (1/T) * sum (V(s_t) - R_t)^2
        actor_loss_accum = 0.0
        critic_loss_accum = 0.0
        for t in range(T):
            p = max(eps, action_probabilities[t])
            actor_loss_accum += math.log(p) * gae_advantages[t]
            diff = state_values[t] - value_targets[t]
            critic_loss_accum += diff * diff

        actor_loss = - (actor_loss_accum / float(T))
        critic_loss = critic_loss_accum / float(T)

        return {
            "td_residuals": deltas,
            "gae_advantages": gae_advantages,
            "value_targets": value_targets,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }
