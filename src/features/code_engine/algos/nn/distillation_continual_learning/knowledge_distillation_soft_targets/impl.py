from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoKnowledgeDistillationSoftTargets:
    """
    ---
    contract:
      algo_id: ALGO-NN-225
      name: NnAlgoKnowledgeDistillationSoftTargets
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - distillation
        - soft_targets
        - dark_knowledge
        - temperature_scaling
        - kl_divergence
      inputs:
        type: object
        required:
          - student_logits
          - teacher_logits
          - true_label
        properties:
          student_logits:
            type: array
            items:
              type: number
            description: Unnormalized prediction logits of the student model of dimension C.
          teacher_logits:
            type: array
            items:
              type: number
            description: Unnormalized prediction logits of the teacher model of dimension C.
          true_label:
            type: integer
            description: Ground-truth class index in range [0, C-1].
          temperature:
            type: number
            default: 2.0
            description: Softmax temperature T > 1.0 for dark knowledge distillation.
          alpha:
            type: number
            default: 0.5
            description: Weight coefficient for distillation loss in [0.0, 1.0].
      outputs:
        type: object
        required:
          - total_loss
          - distillation_kl_loss
          - hard_ce_loss
          - student_soft_probabilities
          - teacher_soft_probabilities
          - student_pred_class
          - teacher_pred_class
        properties:
          total_loss:
            type: number
            description: Weighted combined distillation loss alpha * T^2 * KL + (1 - alpha) * CE.
          distillation_kl_loss:
            type: number
            description: Scaled Kullback-Leibler divergence T^2 * KL(P_teacher_T || P_student_T).
          hard_ce_loss:
            type: number
            description: Standard cross-entropy loss against hard ground-truth label at T=1.
          student_soft_probabilities:
            type: array
            items:
              type: number
            description: Temperature-scaled student probability distribution.
          teacher_soft_probabilities:
            type: array
            items:
              type: number
            description: Temperature-scaled teacher probability distribution.
          student_pred_class:
            type: integer
            description: Argmax class of student logits.
          teacher_pred_class:
            type: integer
            description: Argmax class of teacher logits.
      complexity:
        time: O(C)
        space: O(C)
      preconditions:
        - len(input.student_logits) > 0
        - len(input.student_logits) == len(input.teacher_logits)
        - 0 <= input.true_label < len(input.student_logits)
        - input.temperature > 0.0
        - 0.0 <= input.alpha <= 1.0
      postconditions:
        - output.total_loss >= 0.0
        - output.distillation_kl_loss >= -1e-6
        - output.hard_ce_loss >= 0.0
        - len(output.student_soft_probabilities) == len(input.student_logits)
        - abs(sum(output.student_soft_probabilities) - 1.0) < 1e-5
        - abs(sum(output.teacher_soft_probabilities) - 1.0) < 1e-5
      certificate: "Total loss equals alpha * T^2 * KL(teacher || student) + (1 - alpha) * CE(student, label)"
      compatible_adapters:
        - ADAPTER-LOGIT-DISTILLATION
        - ADAPTER-DARK-KNOWLEDGE-TUNER
      related_algos:
        - ALGO-NN-208
        - ALGO-NN-226
        - ALGO-NN-211
      references:
        - "https://arxiv.org/abs/1503.02531"
        - "https://arxiv.org/abs/2106.05237"
    ---
    """

    @classmethod
    def _softmax(cls, logits: List[float], temp: float = 1.0) -> List[float]:
        scaled = [x / temp for x in logits]
        max_s = max(scaled)
        exp_s = [math.exp(x - max_s) for x in scaled]
        sum_exp = sum(exp_s)
        return [e / sum_exp for e in exp_s]

    @classmethod
    def forward(
        cls,
        student_logits: List[float],
        teacher_logits: List[float],
        true_label: int,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        c = len(student_logits)
        if c == 0:
            raise ValueError("Precondition failed: logits must not be empty")
        if len(teacher_logits) != c:
            raise ValueError("Precondition failed: student and teacher logits length must match")
        if not (0 <= true_label < c):
            raise ValueError("Precondition failed: true_label must be in [0, C-1]")
        if temperature <= 0.0:
            raise ValueError("Precondition failed: temperature must be positive")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Precondition failed: alpha must be in [0.0, 1.0]")

        # 1. Soft probabilities at temperature T
        p_student = cls._softmax(student_logits, temperature)
        p_teacher = cls._softmax(teacher_logits, temperature)

        # 2. KL Divergence: sum_i p_t(i) * log(p_t(i) / p_s(i))
        kl_div = 0.0
        for i in range(c):
            t_prob = max(1e-12, p_teacher[i])
            s_prob = max(1e-12, p_student[i])
            kl_div += t_prob * math.log(t_prob / s_prob)

        distill_loss = (temperature ** 2) * max(0.0, kl_div)

        # 3. Hard CE loss at T = 1.0
        p_student_hard = cls._softmax(student_logits, 1.0)
        hard_prob = max(1e-12, p_student_hard[true_label])
        ce_loss = -math.log(hard_prob)

        # 4. Total weighted loss
        total_loss = alpha * distill_loss + (1.0 - alpha) * ce_loss

        student_pred = student_logits.index(max(student_logits))
        teacher_pred = teacher_logits.index(max(teacher_logits))

        return {
            "total_loss": total_loss,
            "distillation_kl_loss": distill_loss,
            "hard_ce_loss": ce_loss,
            "student_soft_probabilities": p_student,
            "teacher_soft_probabilities": p_teacher,
            "student_pred_class": student_pred,
            "teacher_pred_class": teacher_pred,
        }
