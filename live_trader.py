"""Persistent paper/live trading loop for promoted production models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import List

import numpy as np
from stable_baselines3 import PPO

from brokers import OrderRequest, create_broker
from improved_hybrid_env import ImprovedHybridTradingEnv
from production_config import load_trading_config, TradingConfig
from risk_manager import RiskManager

ACTIONS = {0: "hold", 1: "buy", 2: "sell"}


class LiveTradingService:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.config.validate()
        if self.config.mode == "live" and self.config.broker == "paper":
            raise ValueError("TRADING_MODE=live requires a non-paper BROKER")
        if self.config.mode == "paper" and self.config.broker != "paper":
            raise ValueError("Use TRADING_MODE=live and ENABLE_LIVE_TRADING=true for real brokers")
        self.broker = create_broker(config.broker, config.state_path, config.ledger_path)
        self.risk = RiskManager(config)
        self.model = PPO.load(config.model_path, device="cpu")
        self.env = ImprovedHybridTradingEnv(
            data_dir=config.data_dir,
            use_ml_signals=True,
            ml_feedback=True,
            max_episode_steps=500,
        )

    def _append_event(self, payload: dict) -> None:
        path = Path(self.config.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _observation_at_latest_bar(self):
        self.env.current_step = max(self.env.window_size, self.env.length - 1)
        return self.env._get_observation()

    def build_order_requests(self, actions: np.ndarray) -> List[OrderRequest]:
        orders: List[OrderRequest] = []
        for pair, action in zip(self.env.pairs, actions):
            if pair not in self.config.pairs:
                continue
            side = ACTIONS.get(int(action), "hold")
            if side == "hold":
                continue
            price = float(self.env.data[pair].iloc[self.env.current_step]["4. close"])
            orders.append(
                OrderRequest(
                    pair=pair,
                    side=side,
                    units=self.config.order_units,
                    price=price,
                    reason="ppo_production_action",
                )
            )
        return orders

    def run_cycle(self) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        if self.risk.kill_switch_active():
            event = {"timestamp": timestamp, "event": "kill_switch_active"}
            self._append_event(event)
            return event
        obs = self._observation_at_latest_bar()
        action, _ = self.model.predict(obs, deterministic=True)
        orders = self.build_order_requests(np.asarray(action))
        positions = self.broker.get_positions()
        accepted, rejected = self.risk.filter_orders(orders, positions)
        results = [asdict(self.broker.place_order(order)) for order in accepted]
        event = {
            "timestamp": timestamp,
            "event": "trading_cycle",
            "mode": self.config.mode,
            "broker": self.broker.name,
            "actions": {pair: ACTIONS.get(int(a), "hold") for pair, a in zip(self.env.pairs, action)},
            "accepted_orders": [asdict(order) for order in accepted],
            "rejected_orders": [
                {"order": asdict(order), "reason": decision.reason}
                for order, decision in rejected
            ],
            "results": results,
        }
        self._append_event(event)
        return event

    def run_forever(self) -> None:
        cycle = 0
        while True:
            cycle += 1
            event = self.run_cycle()
            print(json.dumps(event, indent=2, sort_keys=True))
            if self.config.max_cycles and cycle >= self.config.max_cycles:
                break
            time.sleep(self.config.trade_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production paper/live trading loop")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()
    config = load_trading_config()
    service = LiveTradingService(config)
    if args.once:
        print(json.dumps(service.run_cycle(), indent=2, sort_keys=True))
    else:
        service.run_forever()


if __name__ == "__main__":
    main()
