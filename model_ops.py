"""Model evaluation and promotion utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from improved_hybrid_env import ImprovedHybridTradingEnv
from production_config import load_model_gate_config


def evaluate_model(model_path: str, episodes: int = 10, report_path: str = "reports/model_evaluation.json") -> Dict[str, float]:
    model = PPO.load(model_path, device="cpu")
    env = ImprovedHybridTradingEnv(
        window_size=24, use_ml_signals=True, ml_feedback=True, max_episode_steps=500
    )
    rewards: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < env.max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step += 1
        rewards.append(total_reward)
    report = {
        "model_path": model_path,
        "episodes": episodes,
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "best_reward": float(max(rewards)) if rewards else 0.0,
        "worst_reward": float(min(rewards)) if rewards else 0.0,
        "win_rate": float(len([r for r in rewards if r > 0]) / len(rewards) * 100) if rewards else 0.0,
        "rewards": rewards,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def promotion_passes(report: Dict[str, float], min_average_reward: float, min_win_rate: float) -> bool:
    return (
        float(report.get("average_reward", 0.0)) >= min_average_reward
        and float(report.get("win_rate", 0.0)) >= min_win_rate
    )


def promote_model(candidate_path: str, production_path: str, report_path: str, force: bool = False) -> bool:
    gate = load_model_gate_config()
    report = json.loads(Path(report_path).read_text())
    passed = force or promotion_passes(report, gate.min_average_reward, gate.min_win_rate)
    if not passed:
        print("Promotion rejected by evaluation gate")
        print(json.dumps(report, indent=2, sort_keys=True))
        return False
    src = Path(candidate_path)
    dst = Path(production_path)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_suffix(dst.suffix + ".bak")
        shutil.copy2(dst, backup)
    shutil.copy2(src, dst)
    print(f"Promoted {src} -> {dst}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate or promote model artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", default=load_model_gate_config().candidate_model_path)
    eval_parser.add_argument("--episodes", type=int, default=load_model_gate_config().min_episodes)
    eval_parser.add_argument("--report", default=load_model_gate_config().evaluation_report_path)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--candidate", default=load_model_gate_config().candidate_model_path)
    promote_parser.add_argument("--production", default=load_model_gate_config().production_model_path)
    promote_parser.add_argument("--report", default=load_model_gate_config().evaluation_report_path)
    promote_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        report = evaluate_model(args.model, episodes=args.episodes, report_path=args.report)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "promote":
        raise SystemExit(0 if promote_model(args.candidate, args.production, args.report, args.force) else 1)


if __name__ == "__main__":
    main()
