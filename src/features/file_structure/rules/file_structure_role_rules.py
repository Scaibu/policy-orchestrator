from typing import Dict

class RoleLayerRegistry:
    _registry: Dict[str, str] = {
        "domainservice": "DomainCore",
        "schemacl": "DomainCore",
        "domaintypes": "DomainCore",
        "contextboundary": "DomainCore",
        "publicfacade": "DomainCore",
        "ingressrouter": "IngressDelivery",
        "ingresshandler": "IngressDelivery",
        "routerules": "IngressDelivery",
        "graphqlresolver": "IngressDelivery",
        "grpchandler": "IngressDelivery",
        "eventconsumer": "IngressDelivery",
        "ruleset": "Behavioral",
        "statemachine": "Behavioral",
        "workflowengine": "Behavioral",
        "repositoryport": "Persistence",
        "repositoryadapter": "Persistence",
        "namedquery": "Persistence",
        "databasemigration": "Persistence",
        "schemalock": "Persistence",
        "databasepool": "Persistence",
        "contract": "Contracts",
        "contractgraphql": "Contracts",
        "contractgrpc": "Contracts",
        "contractasyncapi": "Contracts",
        "eventproducer": "Messaging",
        "topicspec": "Messaging",
        "topicslock": "Messaging",
        "deadletterqueue": "Messaging",
        "sharedlibrary": "SharedInfrastructure",
        "envconfigschema": "SharedInfrastructure",
        "opentelemetrytracer": "SharedInfrastructure",
        "testsuite": "SharedInfrastructure",
    }
    _default: str = "DomainCore"

    @classmethod
    def resolve(cls, role: str) -> str:
        return cls._registry.get(role.lower(), cls._default)
