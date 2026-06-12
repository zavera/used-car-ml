# Copyright (c) 2026 Callisto Tech — see LICENSE
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PriceEstimate:
    source: str
    price: float
    count: int
    available: bool = True
    note: str = ""


class PricingAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_estimate(self, vehicle: dict) -> PriceEstimate: ...
