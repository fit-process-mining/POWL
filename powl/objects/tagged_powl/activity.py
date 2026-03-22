from __future__ import annotations

from typing import Any, Optional

from .base import TaggedPOWL
from .types import ModelType


class Activity(TaggedPOWL):
    __slots__ = ("label", "organization", "role")

    def __init__(
        self,
        label: Optional[str] = None,
        organization: Optional[str] = None,
        role: Optional[str] = None,
        min_freq: int = 1,
        max_freq: Optional[int] = 1,
    ) -> None:
        """
        label = None  -> silent (τ) activity
        label = str   -> observable activity
        """
        super().__init__(ModelType.Activity, min_freq=min_freq, max_freq=max_freq)
        if label is not None and not isinstance(label, str):
            raise TypeError(f"label must be str or None, got {type(label).__name__}")
        self.label = label
        self.organization = organization
        self.role = role

    def is_silent(self) -> bool:
        """Return True iff this activity is silent (τ)."""
        return self.label is None

    # ---------- representation ----------
    def pretty(self) -> str:
        lbl = "τ" if self.is_silent() else self.label
        return f"Activity({lbl}, min={self.min_freq}, max={self.max_freq})"

    def __repr__(self) -> str:
        return self.pretty()

    # ---------- cloning / equality ----------
    def clone(self, *, deep: bool = True) -> "Activity":
        return Activity(
            label=self.label,
            organization=self.organization,
            role=self.role,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        )

    def normalize(self) -> "Activity":
        return self.clone(deep=True)

    def same_structure(self, other: object) -> bool:
        return (
            isinstance(other, Activity)
            and self.same_signature(other)
            and self.label == other.label
            and self.organization == other.organization
            and self.role == other.role
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.model_type.value,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "label": self.label,  # None => silent
            "organization": self.organization,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Activity":
        return cls(
            label=data.get("label", None),
            organization=data.get("organization"),
            role=data.get("role"),
            min_freq=int(data.get("min_freq", 1)),
            max_freq=data.get("max_freq", 1),
        )
