from typing import List, Callable, Any, Dict
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class InspectionWorkflowStep:
    def __init__(self, step_id: str, action: Callable[[TraceContext, Dict[str, Any]], Result[Dict[str, Any], Exception]]) -> None:
        self.step_id: str = step_id
        self._action = action

    def execute(self, ctx: TraceContext, state: Dict[str, Any]) -> Result[Dict[str, Any], Exception]:
        return self._action(ctx, state)


class RemoteInspectionWorkflow:
    def __init__(self, steps: List[InspectionWorkflowStep]) -> None:
        self._steps: List[InspectionWorkflowStep] = list(steps)

    def run(self, ctx: TraceContext, initial_state: Dict[str, Any]) -> Result[Dict[str, Any], Exception]:
        current = dict(initial_state)
        for step in self._steps:
            res = step.execute(ctx, current)
            if res.is_err:
                return res
            current.update(res.unwrap())
        return Result.ok(current)
