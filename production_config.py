"""
Production configuration for live/paper trading and model promotion.

All defaults are intentionally conservative. Real-money trading requires both
TRADING_MODE=live and ENABLE_LIVE_TRADING=true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _pairs() -> List[str]:
    raw = os.getenv("TRADING_PAIRS", "XAUUSD,GBPUSD,USDJPY,AUDUSD")
    return [pair.strip().upper() for pair in raw.split(",") if pair.strip()]


@dataclass(frozen=True)
class TradingConfig:
    mode: str = os.getenv("TRADING_MODE", "paper").strip().lower()
    broker: str = os.getenv("BROKER", "paper").strip().lower()
    enable_live_trading: bool = _bool("ENABLE_LIVE_TRADING", False)
    pairs: List[str] = field(default_factory=_pairs)
    trade_interval_seconds: int = _int("TRADE_INTERVAL_SECONDS", 3600)
    max_cycles: int = _int("MAX_TRADING_CYCLES", 0)
    model_path: str = os.getenv(
        "PRODUCTION_MODEL_PATH", "models/improved_hybrid_trading_agent.zip"
    )
    data_dir: str = os.getenv("TRADING_DATA_DIR", "historical_data")
    order_units: float = _float("ORDER_UNITS", 1.0)
    max_position_units: float = _float("MAX_POSITION_UNITS", 3.0)
    max_orders_per_cycle: int = _int("MAX_ORDERS_PER_CYCLE", 4)
    max_daily_loss: float = _float("MAX_DAILY_LOSS", 50.0)
    min_action_confidence: float = _float("MIN_ACTION_CONFIDENCE", 0.0)
    kill_switch_file: str = os.getenv("KILL_SWITCH_FILE", "STOP_TRADING")
    ledger_path: str = os.getenv("TRADE_LEDGER_PATH", "logs/trade_ledger.jsonl")
    state_path: str = os.getenv("PAPER_BROKER_STATE", "logs/paper_broker_state.json")

    @property
    def live_enabled(self) -> bool:
        return self.mode == "live" and self.enable_live_trading

    def validate(self) -> None:
        if self.mode not in {"paper", "live"}:
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        if self.mode == "live" and not self.enable_live_trading:
            raise ValueError(
                "Live mode requested, but ENABLE_LIVE_TRADING is not true. "
                "This safety gate prevents accidental real-money trading."
            )
        if self.order_units <= 0:
            raise ValueError("ORDER_UNITS must be positive")
        if self.max_position_units < self.order_units:
            raise ValueError("MAX_POSITION_UNITS must be >= ORDER_UNITS")
        Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelGateConfig:
    candidate_model_path: str = os.getenv(
        "CANDIDATE_MODEL_PATH", "models/candidates/improved_hybrid_trading_agent.zip"
    )
    production_model_path: str = os.getenv(
        "PRODUCTION_MODEL_PATH", "models/improved_hybrid_trading_agent.zip"
    )
    evaluation_report_path: str = os.getenv(
        "EVALUATION_REPORT_PATH", "reports/model_evaluation.json"
    )
    min_average_reward: float = _float("MIN_PROMOTION_AVG_REWARD", 0.0)
    min_win_rate: float = _float("MIN_PROMOTION_WIN_RATE", 40.0)
    min_episodes: int = _int("MODEL_EVAL_EPISODES", 10)


def load_trading_config() -> TradingConfig:
    config = TradingConfig()
    config.validate()
    return config


def load_model_gate_config() -> ModelGateConfig:
    return ModelGateConfig()
