from typing import Generic, TypeVar, Union, Optional

T = TypeVar("T")
E = TypeVar("E")


class Result(Generic[T, E]):
    def __init__(self, is_ok: bool, value: Optional[T] = None, error: Optional[E] = None) -> None:
        self._is_ok: bool = is_ok
        self._value: Optional[T] = value
        self._error: Optional[E] = error

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        return cls(is_ok=True, value=value, error=None)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        return cls(is_ok=False, value=None, error=error)

    @property
    def is_ok(self) -> bool:
        return self._is_ok

    @property
    def is_err(self) -> bool:
        return not self._is_ok

    def unwrap(self) -> T:
        if not self._is_ok:
            raise RuntimeError(f"Called unwrap on Result.err: {self._error}")
        return self._value  # type: ignore

    def error(self) -> E:
        if self._is_ok:
            raise RuntimeError(f"Called error on Result.ok: {self._value}")
        return self._error  # type: ignore
