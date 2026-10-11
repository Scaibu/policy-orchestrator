from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoTwinDelayedDdpgTd3:
    """
    ---
    contract:
      algo_id: ALGO-NN-202
      name: NnAlgoTwinDelayedDdpgTd3
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - td3
        - ddpg
        - continuous_control
        - clipped_double_q
      inputs:
        type: object
        required:
          - current_q1_values
          - current_q2_values
          - target_q1_values
          - target_q2_values
          - rewards
          - dones
          - actor_q1_values
          - target_action_noise
        properties:
          current_q1_values:
            type: array
            items:
              type: number
            description: Online critic 1 evaluations Q1(s, a).
          current_q2_values:
            type: array
            items:
              type: number
            description: Online critic 2 evaluations Q2(s, a).
          target_q1_values:
            type: array
            items:
              type: number
            description: Target critic 1 evaluations on target action Q1_targ(s', a_tilde).
          target_q2_values:
            type: array
            items:
              type: number
            description: Target critic 2 evaluations on target action Q2_targ(s', a_tilde).
          rewards:
            type: array
            items:
              type: number
            description: Transition scalar rewards r.
          dones:
            type: array
            items:
              type: boolean
            description: Episode termination flags.
          actor_q1_values:
            type: array
            items:
              type: number
            description: Critic 1 evaluations on deterministic actor actions Q1(s, mu(s)).
          target_action_noise:
            type: array
            items:
              type: number
            description: Raw exploration noise samples epsilon before clipping.
          gamma:
            type: number
            default: 0.99
            description: Discount factor gamma.
          noise_clip:
            type: number
            default: 0.5
            description: Clipping boundary c for target policy smoothing noise.
          policy_delay:
            type: integer
            default: 2
            description: Frequency interval d for updating actor and target networks.
          current_step:
            type: integer
            default: 1
            description: Current training iteration step.
      outputs:
        type: object
        required:
          - td_target
          - critic_loss
          - actor_loss
          - actor_updated
          - effective_noise
        properties:
          td_target:
            type: array
            items:
              type: number
            description: Clipped double-Q Bellman targets r + gamma * (1 - done) * min(Q1_targ, Q2_targ).
          critic_loss:
            type: number
            description: Combined mean squared error for twin critics.
          actor_loss:
            type: number
            description: Deterministic policy gradient loss -mean(Q1(s, mu(s))) if updated, else 0.0.
          actor_updated:
            type: boolean
            description: Flag indicating whether actor update occurred at this step.
          effective_noise:
            type: array
            items:
              type: number
            description: Target policy noise clamped within [-noise_clip, noise_clip].
      complexity:
        time: O(N)
        space: O(N)
      preconditions:
        - len(input.current_q1_values) > 0
        - len(input.current_q2_values) == len(input.current_q1_values)
        - len(input.target_q1_values) == len(input.current_q1_values)
        - len(input.target_q2_values) == len(input.current_q1_values)
        - len(input.rewards) == len(input.current_q1_values)
        - len(input.dones) == len(input.current_q1_values)
        - len(input.actor_q1_values) == len(input.current_q1_values)
        - len(input.target_action_noise) == len(input.current_q1_values)
        - 0.0 < input.gamma <= 1.0
        - input.noise_clip > 0.0
        - input.policy_delay >= 1
      postconditions:
        - len(output.td_target) == len(input.rewards)
        - len(output.effective_noise) == len(input.target_action_noise)
      certificate: "If current_step % policy_delay != 0, actor_updated is False and actor_loss is 0.0"
      compatible_adapters:
        - ADAPTER-CONTINUOUS-CONTROL-TRAINER
        - ADAPTER-DETERMINISTIC-ACTOR-CRITIC
      related_algos:
        - ALGO-NN-198
        - ALGO-NN-201
      references:
        - "https://arxiv.org/abs/1802.09477"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_q1_values: List[float],
        current_q2_values: List[float],
        target_q1_values: List[float],
        target_q2_values: List[float],
        rewards: List[float],
        dones: List[bool],
        actor_q1_values: List[float],
        target_action_noise: List[float],
        gamma: float = 0.99,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        current_step: int = 1,
    ) -> Dict[str, Any]:
        N = len(current_q1_values)
        if N == 0:
            raise ValueError("Precondition failed: current_q1_values must not be empty")
        if (
            len(current_q2_values) != N
            or len(target_q1_values) != N
            or len(target_q2_values) != N
            or len(rewards) != N
            or len(dones) != N
            or len(actor_q1_values) != N
            or len(target_action_noise) != N
        ):
            raise ValueError("Precondition failed: all input lists must have length N")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("Precondition failed: gamma must be in (0, 1]")
        if noise_clip <= 0.0:
            raise ValueError("Precondition failed: noise_clip must be positive")
        if policy_delay < 1:
            raise ValueError("Precondition failed: policy_delay must be at least 1")

        effective_noise = [
            max(-noise_clip, min(noise_clip, noise)) for noise in target_action_noise
        ]

        targets: List[float] = []
        c1_err = 0.0
        c2_err = 0.0

        for i in range(N):
            if dones[i]:
                y = rewards[i]
            else:
                min_target_q = min(target_q1_values[i], target_q2_values[i])
                y = rewards[i] + gamma * min_target_q
            targets.append(y)
            d1 = current_q1_values[i] - y
            d2 = current_q2_values[i] - y
            c1_err += d1 * d1
            c2_err += d2 * d2

        critic_loss = (c1_err + c2_err) / N

        actor_updated = (current_step % policy_delay == 0)
        if actor_updated:
            actor_loss = -sum(actor_q1_values) / N
        else:
            actor_loss = 0.0

        return {
            "td_target": targets,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "actor_updated": actor_updated,
            "effective_noise": effective_noise,
        }
