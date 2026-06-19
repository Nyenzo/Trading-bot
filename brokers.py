"""Broker adapters for paper trading and guarded live execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests


@dataclass
class OrderRequest:
    pair: str
    side: str
    units: float
    reason: str
    price: Optional[float] = None


@dataclass
class OrderResult:
    accepted: bool
    broker: str
    pair: str
    side: str
    units: float
    message: str
    order_id: Optional[str] = None
    timestamp: str = ""


class Broker(ABC):
    name = "base"

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, state_path: str, ledger_path: str):
        self.state_path = Path(state_path)
        self.ledger_path = Path(ledger_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Dict[str, float]]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"positions": {}, "realized_pnl": {}}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True))

    def _append_ledger(self, payload: dict) -> None:
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def get_positions(self) -> Dict[str, float]:
        return dict(self._state.get("positions", {}))

    def place_order(self, request: OrderRequest) -> OrderResult:
        now = datetime.now(timezone.utc).isoformat()
        signed_units = request.units if request.side == "buy" else -request.units
        positions = self._state.setdefault("positions", {})
        positions[request.pair] = float(positions.get(request.pair, 0.0)) + signed_units
        order_id = f"paper-{now}-{request.pair}-{request.side}"
        result = OrderResult(
            accepted=True,
            broker=self.name,
            pair=request.pair,
            side=request.side,
            units=request.units,
            message="paper order recorded",
            order_id=order_id,
            timestamp=now,
        )
        self._append_ledger({"request": asdict(request), "result": asdict(result)})
        self._save_state()
        return result


class OandaBroker(Broker):
    """Minimal OANDA adapter. Requires explicit live enablement outside this class."""

    name = "oanda"

    INSTRUMENTS = {
        "GBPUSD": "GBP_USD",
        "AUDUSD": "AUD_USD",
        "USDJPY": "USD_JPY",
        "XAUUSD": "XAU_USD",
    }

    def __init__(self):
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        env = os.getenv("OANDA_ENV", "practice").strip().lower()
        host = "api-fxpractice.oanda.com" if env != "live" else "api-fxtrade.oanda.com"
        self.base_url = f"https://{host}/v3"
        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID are required for OANDA")

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _instrument(self, pair: str) -> str:
        try:
            return self.INSTRUMENTS[pair]
        except KeyError as exc:
            raise ValueError(f"Unsupported OANDA pair: {pair}") from exc

    def get_positions(self) -> Dict[str, float]:
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"
        response = requests.get(url, headers=self.headers, timeout=20)
        response.raise_for_status()
        positions = {}
        for item in response.json().get("positions", []):
            instrument = item.get("instrument")
            pair = next((p for p, inst in self.INSTRUMENTS.items() if inst == instrument), instrument)
            long_units = float(item.get("long", {}).get("units", 0) or 0)
            short_units = float(item.get("short", {}).get("units", 0) or 0)
            positions[pair] = long_units + short_units
        return positions

    def place_order(self, request: OrderRequest) -> OrderResult:
        now = datetime.now(timezone.utc).isoformat()
        units = request.units if request.side == "buy" else -request.units
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": self._instrument(request.pair),
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        response = requests.post(url, headers=self.headers, json=payload, timeout=20)
        ok = response.status_code < 300
        data = response.json() if response.text else {}
        order_id = data.get("orderCreateTransaction", {}).get("id")
        return OrderResult(
            accepted=ok,
            broker=self.name,
            pair=request.pair,
            side=request.side,
            units=request.units,
            message=response.text[:500],
            order_id=order_id,
            timestamp=now,
        )


def create_broker(name: str, state_path: str, ledger_path: str) -> Broker:
    normalized = name.strip().lower()
    if normalized == "paper":
        return PaperBroker(state_path=state_path, ledger_path=ledger_path)
    if normalized == "oanda":
        return OandaBroker()
    raise ValueError(f"Unsupported broker: {name}")
