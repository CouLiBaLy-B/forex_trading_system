from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import dataclass
from .registry import IndicatorRegistry

class PriceSeries(Protocol):
    def close(self) -> list[float]: ...
    def high(self) -> list[float]: ...
    def low(self) -> list[float]: ...
    def open(self) -> list[float]: ...
    def volume(self) -> list[float]: ...

@dataclass
class IndicatorResult:
    name: str
    values: list[float | None]
    timestamp: list[str]

class TechnicalIndicator(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}
        self.values: list[float | None] = []

    @abstractmethod
    def compute(self, prices: PriceSeries) -> IndicatorResult: ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"

class IndicatorRegistry:
    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        from .sma import SMA
        from .ema import EMA
        from .rsi import RSI
        from .macd import MACD
        from .bollinger import BollingerBands
        from .atr import ATR
        self.register("sma", SMA)
        self.register("ema", EMA)
        self.register("rsi", RSI)
        self.register("macd", MACD)
        self.register("bollinger", BollingerBands)
        self.register("atr", ATR)

    def register(self, name: str, cls):
        self._registry[name] = cls

    def get(self, name: str, params: dict = None):
        if name not in self._registry:
            raise ValueError(f"Unknown indicator: {name}")
        cls = self._registry[name]
        return cls(params or {})

    def get_all(self):
        return self._registry
