from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSoftActorCritic:
    """
    ---
    contract:
      algo_id: ALGO-NN-201
      name: NnAlgoSoftActorCritic
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - soft_actor_critic
        - maximum_entropy_rl
        - continuous_control
      inputs:
        type: object
        required:
          - current_q1_values
          - current_q2_values
          - next_q1_target_values
          - next_q2_target_values
          - rewards
          - next_log_probs
          - dones
          - policy_log_probs
          - policy_q1_values
          - policy_q2_values
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
          next_q1_target_values:
            type: array
            items:
              type: number
            description: Target critic 1 evaluations Q1_targ(s', a').
          next_q2_target_values:
            type: array
            items:
              type: number
            description: Target critic 2 evaluations Q2_targ(s', a').
          rewards:
            type: array
            items:
              type: number
            description: Transition scalar rewards r.
          next_log_probs:
            type: array
            items:
              type: number
            description: Sampled next action log probabilities log pi(a' | s').
          dones:
            type: array
            items:
              type: boolean
            description: Terminal flags for transitions.
          policy_log_probs:
            type: array
            items:
              type: number
            description: Current policy log probabilities for reparameterized actions log pi(a_tilde | s).
          policy_q1_values:
            type: array
            items:
              type: number
            description: Online critic 1 evaluations for reparameterized actions Q1(s, a_tilde).
          policy_q2_values:
            type: array
            items:
              type: number
            description: Online critic 2 evaluations for reparameterized actions Q2(s, a_tilde).
          alpha:
            type: number
            default: 0.2
            description: Temperature parameter balancing reward and entropy.
          gamma:
            type: number
            default: 0.99
            description: Discount factor gamma.
          target_entropy:
            type: number
            default: -1.0
            description: Target minimum entropy H_target (typically -dim(A)).
          alpha_lr:
            type: number
            default: 0.001
            description: Learning rate for dual gradient descent temperature adaptation.
      outputs:
        type: object
        required:
          - critic_target
          - q1_loss
          - q2_loss
          - critic_loss
          - actor_loss
          - alpha_loss
          - updated_alpha
          - mean_entropy
        properties:
          critic_target:
            type: array
            items:
              type: number
            description: Soft Bellman target values r + gamma * (1 - done) * (min(Q1, Q2) - alpha * log pi).
          q1_loss:
            type: number
            description: Mean squared Bellman error for critic 1.
          q2_loss:
            type: number
            description: Mean squared Bellman error for critic 2.
          critic_loss:
            type: number
            description: Sum of critic 1 and critic 2 losses.
          actor_loss:
            type: number
            description: Policy loss mean(alpha * log pi - min(Q1, Q2)).
          alpha_loss:
            type: number
            description: Dual objective loss for entropy temperature tuning.
          updated_alpha:
            type: number
            description: Updated temperature alpha after gradient step.
          mean_entropy:
            type: number
            description: Empirical policy entropy -mean(log pi).
      complexity:
        time: O(N)
        space: O(N)
      preconditions:
        - len(input.current_q1_values) > 0
        - len(input.current_q2_values) == len(input.current_q1_values)
        - len(input.next_q1_target_values) == len(input.current_q1_values)
        - len(input.next_q2_target_values) == len(input.current_q1_values)
        - len(input.rewards) == len(input.current_q1_values)
        - len(input.next_log_probs) == len(input.current_q1_values)
        - len(input.dones) == len(input.current_q1_values)
        - len(input.policy_log_probs) == len(input.current_q1_values)
        - len(input.policy_q1_values) == len(input.current_q1_values)
        - len(input.policy_q2_values) == len(input.current_q1_values)
        - 0.0 < input.gamma <= 1.0
        - input.alpha > 0.0
      postconditions:
        - len(output.critic_target) == len(input.rewards)
        - output.updated_alpha > 0.0
      certificate: "If dones are all True, critic_target equals rewards exactly, independent of gamma or next Q-values"
      compatible_adapters:
        - ADAPTER-CONTINUOUS-CONTROL-TRAINER
        - ADAPTER-MAX-ENTROPY-RL
      related_algos:
        - ALGO-NN-198
        - ALGO-NN-202
      references:
        - "https://arxiv.org/abs/1801.01290"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_q1_values: List[float],
        current_q2_values: List[float],
        next_q1_target_values: List[float],
        next_q2_target_values: List[float],
        rewards: List[float],
        next_log_probs: List[float],
        dones: List[bool],
        policy_log_probs: List[float],
        policy_q1_values: List[float],
        policy_q2_values: List[float],
        alpha: float = 0.2,
        gamma: float = 0.99,
        target_entropy: float = -1.0,
        alpha_lr: float = 0.001,
    ) -> Dict[str, Any]:
        N = len(current_q1_values)
        if N == 0:
            raise ValueError("Precondition failed: current_q1_values must not be empty")
        if (
            len(current_q2_values) != N
            or len(next_q1_target_values) != N
            or len(next_q2_target_values) != N
            or len(rewards) != N
            or len(next_log_probs) != N
            or len(dones) != N
            or len(policy_log_probs) != N
            or len(policy_q1_values) != N
            or len(policy_q2_values) != N
        ):
            raise ValueError("Precondition failed: all input lists must have identical length N")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("Precondition failed: gamma must be in (0, 1]")
        if alpha <= 0.0:
            raise ValueError("Precondition failed: alpha must be positive")

        targets: List[float] = []
        q1_sq_err = 0.0
        q2_sq_err = 0.0

        for i in range(N):
            if dones[i]:
                y = rewards[i]
            else:
                min_next_q = min(next_q1_target_values[i], next_q2_target_values[i])
                soft_val = min_next_q - alpha * next_log_probs[i]
                y = rewards[i] + gamma * soft_val
            targets.append(y)
            diff1 = current_q1_values[i] - y
            diff2 = current_q2_values[i] - y
            q1_sq_err += diff1 * diff1
            q2_sq_err += diff2 * diff2

        q1_loss = q1_sq_err / N
        q2_loss = q2_sq_err / N
        critic_loss = q1_loss + q2_loss

        actor_loss_accum = 0.0
        entropy_accum = 0.0
        alpha_loss_accum = 0.0

        for i in range(N):
            lp = policy_log_probs[i]
            min_q = min(policy_q1_values[i], policy_q2_values[i])
            actor_loss_accum += (alpha * lp - min_q)
            entropy_accum += (-lp)
            alpha_loss_accum += (-alpha * (lp + target_entropy))

        actor_loss = actor_loss_accum / N
        mean_entropy = entropy_accum / N
        alpha_loss = alpha_loss_accum / N

        # Dual gradient descent update for temperature: d/d(alpha) alpha_loss = - (log pi + target_entropy)
        alpha_grad = - ( -mean_entropy + target_entropy )
        updated_alpha = max(1e-6, alpha - alpha_lr * alpha_grad)

        return {
            "critic_target": targets,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "updated_alpha": updated_alpha,
            "mean_entropy": mean_entropy,
        }
