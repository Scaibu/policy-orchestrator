from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRlVerifiableRewardsRlvr:
    """
    ---
    contract:
      algo_id: ALGO-NN-214
      name: NnAlgoRlVerifiableRewardsRlvr
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - llm_alignment
        - rlvr
        - verifiable_rewards
        - code_generation
        - mathematical_reasoning
      inputs:
        type: object
        required:
          - test_results
          - format_valid
          - response_lengths
        properties:
          test_results:
            type: array
            items:
              type: array
              items:
                type: boolean
            description: Boolean execution test results (Pass/Fail) for each test suite across N candidate responses.
          format_valid:
            type: array
            items:
              type: boolean
            description: Boolean flags indicating whether response adhered to required formatting (e.g. reasoning tags, boxed answers).
          response_lengths:
            type: array
            items:
              type: integer
            description: Token lengths of each response for length penalty computation.
          max_target_length:
            type: integer
            default: 1024
            description: Maximum length threshold beyond which linear truncation penalty applies.
          weight_pass:
            type: number
            default: 1.0
            description: Reward multiplier for passing all tests in the verification suite.
          weight_format:
            type: number
            default: 0.2
            description: Reward bonus for valid syntax and reasoning tag structure.
          weight_length_penalty:
            type: number
            default: 0.0001
            description: Linear penalty per token exceeding max_target_length.
      outputs:
        type: object
        required:
          - total_rewards
          - pass_at_1
          - all_tests_passed
          - mean_format_adherence
        properties:
          total_rewards:
            type: array
            items:
              type: number
            description: Composite scalar verifiable reward for each candidate response.
          pass_at_1:
            type: number
            description: Empirical pass@1 rate across candidate solutions in [0, 1].
          all_tests_passed:
            type: array
            items:
              type: boolean
            description: Boolean flag for each candidate indicating whether all suite unit tests passed.
          mean_format_adherence:
            type: number
            description: Proportion of candidate responses conforming to required output schema.
      complexity:
        time: O(N * M)
        space: O(N)
      preconditions:
        - len(input.test_results) > 0
        - len(input.format_valid) == len(input.test_results)
        - len(input.response_lengths) == len(input.test_results)
        - all(len(suite) > 0 for suite in input.test_results)
        - input.weight_pass >= 0.0
        - input.weight_format >= 0.0
      postconditions:
        - len(output.total_rewards) == len(input.test_results)
        - len(output.all_tests_passed) == len(input.test_results)
        - 0.0 <= output.pass_at_1 <= 1.0
        - 0.0 <= output.mean_format_adherence <= 1.0
      certificate: "If all tests pass, format is valid, and length <= max, total_reward is exactly weight_pass + weight_format"
      compatible_adapters:
        - ADAPTER-RLVR-VERIFIER
        - ADAPTER-UNIT-TEST-REWARD-ENVIRONMENT
      related_algos:
        - ALGO-NN-213
        - ALGO-NN-216
        - ALGO-NN-217
      references:
        - "https://arxiv.org/abs/2402.03300"
        - "https://arxiv.org/abs/2308.01825"
    ---
    """

    @classmethod
    def forward(
        cls,
        test_results: List[List[bool]],
        format_valid: List[bool],
        response_lengths: List[int],
        max_target_length: int = 1024,
        weight_pass: float = 1.0,
        weight_format: float = 0.2,
        weight_length_penalty: float = 0.0001,
    ) -> Dict[str, Any]:
        N = len(test_results)
        if N == 0:
            raise ValueError("Precondition failed: test_results must not be empty")
        if len(format_valid) != N or len(response_lengths) != N:
            raise ValueError("Precondition failed: input lists must have length N")
        if any(len(suite) == 0 for suite in test_results):
            raise ValueError("Precondition failed: each test suite must have at least one test")
        if weight_pass < 0.0 or weight_format < 0.0 or weight_length_penalty < 0.0:
            raise ValueError("Precondition failed: reward weights must be non-negative")

        rewards: List[float] = []
        all_passed: List[bool] = []
        passed_count = 0
        valid_format_count = 0

        for i in range(N):
            suite = test_results[i]
            # Verify if every unit test in the suite passed
            is_full_pass = all(suite)
            all_passed.append(is_full_pass)
            if is_full_pass:
                passed_count += 1

            pass_score = weight_pass if is_full_pass else 0.0

            # Format bonus
            is_fmt = format_valid[i]
            if is_fmt:
                valid_format_count += 1
            fmt_score = weight_format if is_fmt else 0.0

            # Length penalty
            length = response_lengths[i]
            excess = max(0, length - max_target_length)
            len_penalty = weight_length_penalty * excess

            total_r = pass_score + fmt_score - len_penalty
            rewards.append(total_r)

        pass_rate = passed_count / float(N)
        mean_fmt = valid_format_count / float(N)

        return {
            "total_rewards": rewards,
            "pass_at_1": pass_rate,
            "all_tests_passed": all_passed,
            "mean_format_adherence": mean_fmt,
        }
