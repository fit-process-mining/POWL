from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from .types import ModelType


class TaggedPOWL(ABC):
    """
    Abstract base for all TaggedPOWL models.

    Frequency semantics:
      - min_freq: integer >= 0
      - max_freq: integer >= min_freq OR None (unbounded)

    Design notes:
      - Default equality/hash are identity-based to be safe when these objects
        are used as NetworkX nodes (mutable structures + structural equality
        can get tricky fast).
      - Structural comparison is provided via same_structure().
    """

    __slots__ = ("model_type", "min_freq", "max_freq")

    def __init__(
        self,
        model_type: ModelType,
        min_freq: int = 1,
        max_freq: Optional[int] = 1,
    ) -> None:
        # Prevent direct instantiation of the abstract base even if someone tries.
        if type(self) is TaggedPOWL:
            raise TypeError("TaggedPOWL is abstract and cannot be instantiated directly.")

        self._validate_freqs(min_freq=min_freq, max_freq=max_freq)
        self.model_type: ModelType = model_type
        self.min_freq: int = int(min_freq)
        self.max_freq: Optional[int] = None if max_freq is None else int(max_freq)

    # ---------- required interface ----------
    @abstractmethod
    def clone(self, *, deep: bool = True) -> "TaggedPOWL":
        """Return a copy of this model. If deep=True, graph-backed models clone structure too."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize this node/model into a JSON-friendly dict."""
        raise NotImplementedError

    @abstractmethod
    def pretty(self) -> str:
        """Human-friendly string representation (multi-line allowed)."""
        raise NotImplementedError

    @abstractmethod
    def same_structure(self, other: object) -> bool:
        """Structural equality."""
        raise NotImplementedError

    # ---------- common helpers ----------
    @staticmethod
    def _validate_freqs(min_freq: int, max_freq: Optional[int]) -> None:
        if not isinstance(min_freq, int):
            raise TypeError(f"min_freq must be int, got {type(min_freq).__name__}")
        if min_freq < 0:
            raise ValueError("min_freq must be >= 0")

        if max_freq is None:
            return
        if not isinstance(max_freq, int):
            raise TypeError(f"max_freq must be int or None, got {type(max_freq).__name__}")
        if max_freq < min_freq:
            raise ValueError("max_freq must be >= min_freq (or None for unbounded)")

    @abstractmethod
    def normalize(self) -> "TaggedPOWL":
        raise NotImplementedError

    def set_freqs(self, *, min_freq: int, max_freq: Optional[int]) -> None:
        self._validate_freqs(min_freq=min_freq, max_freq=max_freq)
        self.min_freq = int(min_freq)
        self.max_freq = None if max_freq is None else int(max_freq)

    def is_skippable(self) -> bool:
        return self.min_freq == 0

    def is_repeatable(self) -> bool:
        return self.max_freq is None or self.max_freq > 1

    def is_unbounded(self) -> bool:
        return self.max_freq is None

    def freq_range(self) -> Tuple[int, Optional[int]]:
        return (self.min_freq, self.max_freq)

    def same_signature(self, other: object) -> bool:
        """Shallow compare of the common tag fields (ignores internal structure)."""
        if not isinstance(other, TaggedPOWL):
            return False
        return (
            self.model_type == other.model_type
            and self.min_freq == other.min_freq
            and self.max_freq == other.max_freq
        )

    # ---------- safe defaults for node identity ----------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type.value}, "
            f"min_freq={self.min_freq}, "
            f"max_freq={self.max_freq}"
            f")"
        )

    def __str__(self) -> str:
        return self.pretty()

    def __hash__(self) -> int:
        # Identity hash => safe for NetworkX node usage.
        return id(self)

    def __eq__(self, other: object) -> bool:
        # Identity equality => safe for NetworkX node usage.
        return self is other
