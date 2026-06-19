"""Production risk controls for generated trade requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from brokers import OrderRequest
from production_config import TradingConfig


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config

    def kill_switch_active(self) -> bool:
        return Path(self.config.kill_switch_file).exists()

    def validate_order(self, request: OrderRequest, positions: Dict[str, float]) -> RiskDecision:
        if self.kill_switch_active():
            return RiskDecision(False, f"kill switch active: {self.config.kill_switch_file}")
        if request.pair not in self.config.pairs:
            return RiskDecision(False, f"pair {request.pair} is not enabled")
        if request.units <= 0:
            return RiskDecision(False, "order units must be positive")
        current = float(positions.get(request.pair, 0.0))
        signed = request.units if request.side == "buy" else -request.units
        projected = current + signed
        if abs(projected) > self.config.max_position_units:
            return RiskDecision(
                False,
                f"projected position {projected} exceeds max {self.config.max_position_units}",
            )
        return RiskDecision(True, "allowed")

    def filter_orders(
        self, requests: Iterable[OrderRequest], positions: Dict[str, float]
    ) -> Tuple[List[OrderRequest], List[Tuple[OrderRequest, RiskDecision]]]:
        accepted: List[OrderRequest] = []
        rejected: List[Tuple[OrderRequest, RiskDecision]] = []
        shadow_positions = dict(positions)
        for request in requests:
            if len(accepted) >= self.config.max_orders_per_cycle:
                rejected.append((request, RiskDecision(False, "max orders per cycle reached")))
                continue
            decision = self.validate_order(request, shadow_positions)
            if decision.allowed:
                signed = request.units if request.side == "buy" else -request.units
                shadow_positions[request.pair] = shadow_positions.get(request.pair, 0.0) + signed
                accepted.append(request)
            else:
                rejected.append((request, decision))
        return accepted, rejected
