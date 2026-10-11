from .knowledge_distillation_soft_targets.impl import NnAlgoKnowledgeDistillationSoftTargets
from .sequence_on_policy_distillation.impl import NnAlgoSequenceOnPolicyDistillation
from .elastic_weight_consolidation_ewc.impl import NnAlgoElasticWeightConsolidationEwc
from .experience_replay_continual_learning.impl import NnAlgoExperienceReplayContinualLearning

__all__ = [
    "NnAlgoKnowledgeDistillationSoftTargets",
    "NnAlgoSequenceOnPolicyDistillation",
    "NnAlgoElasticWeightConsolidationEwc",
    "NnAlgoExperienceReplayContinualLearning",
]
