from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSelfDistillationDino:
    """
    ---
    contract:
      algo_id: ALGO-NN-175
      name: NnAlgoSelfDistillationDino
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - self_supervised_learning
        - dino
        - self_distillation
        - teacher_centering
        - sharpening
      inputs:
        type: object
        required:
          - student_logits
          - teacher_logits
          - center_vector
        properties:
          student_logits:
            type: array
            items:
              type: number
            description: Student network output logits of dimension K.
          teacher_logits:
            type: array
            items:
              type: number
            description: Teacher EMA network output logits of dimension K.
          center_vector:
            type: array
            items:
              type: number
            description: Running mean center vector c of dimension K.
          tau_student:
            type: number
            default: 0.1
            description: Student sharpening temperature.
          tau_teacher:
            type: number
            default: 0.04
            description: Teacher sharpening temperature.
          center_momentum:
            type: number
            default: 0.9
            description: Momentum for updating center vector c.
      outputs:
        type: object
        required:
          - cross_entropy_loss
          - updated_center_vector
          - teacher_entropy
          - student_entropy
          - max_teacher_prob
        properties:
          cross_entropy_loss:
            type: number
            description: Cross-entropy loss H(P_teacher, P_student).
          updated_center_vector:
            type: array
            items:
              type: number
            description: Updated running center vector m * c + (1 - m) * teacher_logits.
          teacher_entropy:
            type: number
            description: Shannon entropy of the sharpened centered teacher distribution.
          student_entropy:
            type: number
            description: Shannon entropy of the student distribution.
          max_teacher_prob:
            type: number
            description: Maximum probability in the teacher distribution (confidence indicator).
      parameters: {}
      input_assumptions:
        - student_logits, teacher_logits, and center_vector have matching positive dimension K
        - tau_student > 0.0 and tau_teacher > 0.0
        - 0.0 <= center_momentum <= 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized softmax with max-subtraction and eps 1e-12"
      uses_model: false
      complexity:
        variables:
          K: prototype dimension
        time_worst: O(K)
        time_typical: O(K)
        space: O(K)
      preconditions:
        - len(input.student_logits) > 0
        - len(input.student_logits) == len(input.teacher_logits) == len(input.center_vector)
        - input.tau_student > 0.0 and input.tau_teacher > 0.0
        - 0.0 <= input.center_momentum <= 1.0
      postconditions:
        - output.cross_entropy_loss >= 0.0
        - len(output.updated_center_vector) == len(input.student_logits)
        - output.teacher_entropy >= 0.0
        - 0.0 <= output.max_teacher_prob <= 1.000001
      certificate: "Cross-entropy strictly equals -sum(P_teacher * log(P_student))"
      compatible_adapters:
        - ADAPTER-DINO-PRETRAINER
        - ADAPTER-DINOV2-STUDENT-TEACHER
      related_algos:
        - ALGO-NN-172
        - ALGO-NN-173
        - ALGO-NN-176
      references:
        - "https://arxiv.org/abs/2104.14294"
        - "https://arxiv.org/abs/2304.07193"
    ---
    """

    @staticmethod
    def _stable_softmax(logits: List[float], tau: float) -> List[float]:
        scaled = [x / tau for x in logits]
        max_val = max(scaled)
        exps = [math.exp(x - max_val) for x in scaled]
        sum_exp = sum(exps)
        return [e / sum_exp for e in exps]

    @classmethod
    def forward(
        cls,
        student_logits: List[float],
        teacher_logits: List[float],
        center_vector: List[float],
        tau_student: float = 0.1,
        tau_teacher: float = 0.04,
        center_momentum: float = 0.9,
    ) -> Dict[str, Any]:
        K = len(student_logits)
        if K == 0:
            raise ValueError("Precondition failed: logits cannot be empty")
        if len(teacher_logits) != K or len(center_vector) != K:
            raise ValueError("Precondition failed: dimension mismatch across vectors")
        if tau_student <= 0.0 or tau_teacher <= 0.0:
            raise ValueError("Precondition failed: temperatures must be positive")
        if not (0.0 <= center_momentum <= 1.0):
            raise ValueError("Precondition failed: center_momentum must be in [0, 1]")

        eps = 1e-12

        # 1. Center teacher logits: t_centered = teacher_logits - center_vector
        centered_teacher = [teacher_logits[i] - center_vector[i] for i in range(K)]

        # 2. Sharpen teacher & student distributions:
        p_teacher = cls._stable_softmax(centered_teacher, tau_teacher)
        p_student = cls._stable_softmax(student_logits, tau_student)

        # 3. Cross-entropy loss: H(P_t, P_s) = - sum_k P_t[k] * log(P_s[k] + eps)
        ce_loss = 0.0
        h_teacher = 0.0
        h_student = 0.0

        for k in range(K):
            pt = p_teacher[k]
            ps = p_student[k]
            ce_loss -= pt * math.log(max(eps, ps))

            if pt > eps:
                h_teacher -= pt * math.log(pt)
            if ps > eps:
                h_student -= ps * math.log(ps)

        # 4. Update running center vector: c_new = m * c + (1 - m) * teacher_logits
        updated_center: List[float] = []
        for k in range(K):
            val = center_momentum * center_vector[k] + (1.0 - center_momentum) * teacher_logits[k]
            updated_center.append(val)

        return {
            "cross_entropy_loss": ce_loss,
            "updated_center_vector": updated_center,
            "teacher_entropy": h_teacher,
            "student_entropy": h_student,
            "max_teacher_prob": max(p_teacher),
        }
