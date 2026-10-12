from typing import Dict, Tuple, Callable

class EdgeDirectionRegistry:
    _handlers: Dict[str, Callable[[str, str], Tuple[str, str]]] = {
        "incoming": lambda node_id, target_id: (target_id, node_id),
        "outgoing": lambda node_id, target_id: (node_id, target_id),
    }

    @classmethod
    def resolve(cls, direction: str, node_id: str, target_id: str) -> Tuple[str, str]:
        handler = cls._handlers.get(direction.lower(), cls._handlers["outgoing"])
        return handler(node_id, target_id)
