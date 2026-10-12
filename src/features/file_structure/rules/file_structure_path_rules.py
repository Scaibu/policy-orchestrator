from typing import List, Tuple, Callable

class PathLabelRuleEngine:
    _rules: List[Tuple[Callable[[str], bool], str]] = [
        (lambda p: "contracts/openapi" in p or p.endswith(".yaml") or p.endswith(".json"), "Contract"),
        (lambda p: "router" in p and "routers/" in p, "IngressRouter"),
        (lambda p: "handler" in p, "IngressHandler"),
        (lambda p: "schema" in p, "SchemaACL"),
        (lambda p: "service" in p, "DomainService"),
        (lambda p: "rule" in p, "RuleSet"),
        (lambda p: "machine" in p, "StateMachine"),
        (lambda p: "workflow" in p, "WorkflowEngine"),
        (lambda p: "port" in p, "RepositoryPort"),
        (lambda p: "adapter" in p or "repository" in p, "RepositoryAdapter"),
        (lambda p: "queries" in p or p.endswith(".sql"), "NamedQuery"),
        (lambda p: "migration" in p, "DatabaseMigration"),
    ]
    _default: str = "Unknown"

    @classmethod
    def resolve(cls, path: str) -> str:
        p = path.lower()
        for matcher, label in cls._rules:
            if matcher(p):
                return label
        return cls._default
