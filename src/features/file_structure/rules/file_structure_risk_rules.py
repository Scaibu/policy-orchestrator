from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class RuleCondition:
    field: str
    operator: str
    value: Any

@dataclass(frozen=True)
class RuleEffect:
    effect: str
    value: Any

@dataclass(frozen=True)
class BusinessRule:
    id: str
    name: str
    priority: int
    category: str
    conditions: List[RuleCondition]
    effects: List[RuleEffect]

FILE_STRUCTURE_RULES: List[BusinessRule] = [
    BusinessRule(
        id="rule_contract_boundary_risk",
        name="Contract Boundary Modification High Risk Rule",
        priority=100,
        category="risk_classification",
        conditions=[
            RuleCondition(field="target_label", operator="equals", value="Contract"),
        ],
        effects=[
            RuleEffect(effect="set_risk_level", value="Level 3: Public Contract Boundary Risk"),
            RuleEffect(effect="add_verification_command", value="npm run test:contract"),
        ],
    ),
    BusinessRule(
        id="rule_storage_ddl_risk",
        name="Storage DDL Modification High Risk Rule",
        priority=90,
        category="risk_classification",
        conditions=[
            RuleCondition(field="target_label", operator="in", value=["DatabaseMigration", "NamedQuery"]),
        ],
        effects=[
            RuleEffect(effect="set_risk_level", value="Level 2: Storage & Persistence Migration Risk"),
            RuleEffect(effect="add_verification_command", value="npm run migrate:check && npm run test:integration"),
        ],
    ),
    BusinessRule(
        id="rule_isolated_business_logic_scope",
        name="Isolated Business Logic Rule Scope",
        priority=80,
        category="risk_classification",
        conditions=[
            RuleCondition(field="target_label", operator="in", value=["RuleSet", "StateMachine", "WorkflowEngine"]),
        ],
        effects=[
            RuleEffect(effect="set_risk_level", value="Level 0: Isolated Business Logic Scope"),
            RuleEffect(effect="add_verification_command", value="pytest tests/unit/"),
        ],
    ),
    BusinessRule(
        id="rule_local_feature_scope",
        name="Local Feature Architecture Scope",
        priority=70,
        category="risk_classification",
        conditions=[
            RuleCondition(field="target_label", operator="in", value=["SchemaACL", "RepositoryPort", "DomainModel", "RepositoryAdapter"]),
        ],
        effects=[
            RuleEffect(effect="set_risk_level", value="Level 1: Local Feature Scope"),
            RuleEffect(effect="add_verification_command", value="pytest tests/unit/"),
        ],
    ),
]

class FileStructureRuleSet:
    @staticmethod
    def evaluate_rules(context: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "risk_level": "Level 1: Local Feature Scope",
            "verification_commands": [],
        }

        sorted_rules = sorted(FILE_STRUCTURE_RULES, key=lambda r: r.priority, reverse=True)
        for rule in sorted_rules:
            matched = True
            for cond in rule.conditions:
                ctx_val = context.get(cond.field)
                if cond.operator == "equals" and ctx_val != cond.value:
                    matched = False
                    break
                elif cond.operator == "in" and ctx_val not in cond.value:
                    matched = False
                    break

            if matched:
                for eff in rule.effects:
                    if eff.effect == "set_risk_level":
                        results["risk_level"] = eff.value
                    elif eff.effect == "add_verification_command":
                        if eff.value not in results["verification_commands"]:
                            results["verification_commands"].append(eff.value)
                break

        return results

KnowledgeGraphRuleSet = FileStructureRuleSet
