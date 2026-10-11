from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSequenceOnPolicyDistillation:
    """
    ---
    contract:
      algo_id: ALGO-NN-226
      name: NnAlgoSequenceOnPolicyDistillation
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - distillation
        - sequence_distillation
        - on_policy
        - exposure_bias
        - forward_kl
        - reverse_kl
      inputs:
        type: object
        required:
          - student_sequence_logits
          - teacher_sequence_logits
          - token_ids
        properties:
          student_sequence_logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Student prediction logits across sequence tokens of shape (seq_len, vocab_size).
          teacher_sequence_logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Teacher prediction logits across sequence tokens of shape (seq_len, vocab_size).
          token_ids:
            type: array
            items:
              type: integer
            description: Sequence token IDs of length seq_len.
          distillation_mode:
            type: string
            default: forward_kl
            enum: [forward_kl, reverse_kl, sequence_sft, jsd]
            description: Distillation alignment objective.
          temperature:
            type: number
            default: 1.0
            description: Softmax temperature for probability distribution smoothing.
      outputs:
        type: object
        required:
          - token_losses
          - mean_sequence_loss
          - per_token_student_entropies
          - per_token_teacher_entropies
          - effective_sequence_length
        properties:
          token_losses:
            type: array
            items:
              type: number
            description: Stepwise distillation loss for each position in sequence.
          mean_sequence_loss:
            type: number
            description: Average loss across the sequence length.
          per_token_student_entropies:
            type: array
            items:
              type: number
            description: Shannon entropy of student next-token distribution at each position.
          per_token_teacher_entropies:
            type: array
            items:
              type: number
            description: Shannon entropy of teacher next-token distribution at each position.
          effective_sequence_length:
            type: integer
            description: Number of evaluated sequence tokens.
      complexity:
        time: O(seq_len * vocab_size)
        space: O(seq_len * vocab_size)
      preconditions:
        - len(input.student_sequence_logits) > 0
        - len(input.student_sequence_logits) == len(input.teacher_sequence_logits)
        - len(input.student_sequence_logits) == len(input.token_ids)
        - all(len(s) == len(t) for s, t in zip(input.student_sequence_logits, input.teacher_sequence_logits))
        - input.temperature > 0.0
      postconditions:
        - len(output.token_losses) == len(input.token_ids)
        - output.mean_sequence_loss >= -1e-6
        - all(e >= 0.0 for e in output.per_token_student_entropies)
        - output.effective_sequence_length == len(input.token_ids)
      certificate: "Mean sequence loss equals arithmetic mean of stepwise token distillation losses"
      compatible_adapters:
        - ADAPTER-SEQ-KD-LOSS
        - ADAPTER-ON-POLICY-ALIGNMENT
      related_algos:
        - ALGO-NN-208
        - ALGO-NN-225
        - ALGO-NN-213
      references:
        - "https://arxiv.org/abs/1606.07947"
        - "https://arxiv.org/abs/2306.08543"
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
    def _entropy(cls, probs: List[float]) -> float:
        ent = 0.0
        for p in probs:
            if p > 1e-12:
                ent -= p * math.log(p)
        return max(0.0, ent)

    @classmethod
    def forward(
        cls,
        student_sequence_logits: List[List[float]],
        teacher_sequence_logits: List[List[float]],
        token_ids: List[int],
        distillation_mode: str = "forward_kl",
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        seq_len = len(student_sequence_logits)
        if seq_len == 0:
            raise ValueError("Precondition failed: sequence cannot be empty")
        if len(teacher_sequence_logits) != seq_len or len(token_ids) != seq_len:
            raise ValueError("Precondition failed: sequence lengths must match")
        vocab_size = len(student_sequence_logits[0])
        if vocab_size == 0 or any(len(row) != vocab_size for row in student_sequence_logits):
            raise ValueError("Precondition failed: inconsistent student vocab size")
        if any(len(row) != vocab_size for row in teacher_sequence_logits):
            raise ValueError("Precondition failed: teacher vocab size must match student")
        if any(not (0 <= tid < vocab_size) for tid in token_ids):
            raise ValueError("Precondition failed: token_ids must be within vocab range")
        if temperature <= 0.0:
            raise ValueError("Precondition failed: temperature must be positive")

        token_losses: List[float] = []
        student_entropies: List[float] = []
        teacher_entropies: List[float] = []

        for t in range(seq_len):
            s_logits = student_sequence_logits[t]
            t_logits = teacher_sequence_logits[t]
            tid = token_ids[t]

            p_s = cls._softmax(s_logits, temperature)
            p_t = cls._softmax(t_logits, temperature)

            student_entropies.append(cls._entropy(p_s))
            teacher_entropies.append(cls._entropy(p_t))

            if distillation_mode == "sequence_sft":
                # Supervised fine-tuning on teacher's generated token
                prob = max(1e-12, p_s[tid])
                step_loss = -math.log(prob)
            elif distillation_mode == "reverse_kl":
                # On-policy reverse KL: KL(p_student || p_teacher)
                step_loss = sum(
                    p_s[i] * math.log(max(1e-12, p_s[i]) / max(1e-12, p_t[i]))
                    for i in range(vocab_size)
                )
            elif distillation_mode == "jsd":
                # Jensen-Shannon divergence: 0.5 * (KL(p_t || m) + KL(p_s || m))
                m_dist = [0.5 * (p_t[i] + p_s[i]) for i in range(vocab_size)]
                kl_tm = sum(
                    p_t[i] * math.log(max(1e-12, p_t[i]) / max(1e-12, m_dist[i]))
                    for i in range(vocab_size)
                )
                kl_sm = sum(
                    p_s[i] * math.log(max(1e-12, p_s[i]) / max(1e-12, m_dist[i]))
                    for i in range(vocab_size)
                )
                step_loss = 0.5 * (kl_tm + kl_sm)
            else:
                # Default: forward KL: KL(p_teacher || p_student)
                step_loss = sum(
                    p_t[i] * math.log(max(1e-12, p_t[i]) / max(1e-12, p_s[i]))
                    for i in range(vocab_size)
                )

            token_losses.append(max(0.0, step_loss))

        mean_loss = sum(token_losses) / float(seq_len)

        return {
            "token_losses": token_losses,
            "mean_sequence_loss": mean_loss,
            "per_token_student_entropies": student_entropies,
            "per_token_teacher_entropies": teacher_entropies,
            "effective_sequence_length": seq_len,
        }
