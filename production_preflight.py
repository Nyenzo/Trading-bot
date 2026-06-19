"""Readiness checks for the no-VPS production staging milestone."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import Callable, List

from dotenv import load_dotenv

from production_config import load_trading_config

load_dotenv()


@dataclass
class CheckResult:
    name: str
    status: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "deferred"}


def _exists(path: str) -> bool:
    return Path(path).exists()


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def _env_present(names: List[str]) -> List[str]:
    return [name for name in names if os.getenv(name)]


def _check_python_modules() -> CheckResult:
    missing = [
        module
        for module in ("numpy", "pandas", "sklearn", "stable_baselines3", "torch")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return CheckResult("python_dependencies", "fail", f"Missing modules: {', '.join(missing)}")
    return CheckResult("python_dependencies", "pass", "Core ML/trading modules are importable")


def _check_cpu_torch() -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return CheckResult("cpu_torch", "fail", f"Could not import torch: {exc}")
    if torch.cuda.is_available():
        return CheckResult("cpu_torch", "fail", "CUDA is available; this staging profile expects CPU-only execution")
    return CheckResult("cpu_torch", "pass", f"torch {torch.__version__} is running without CUDA")


def _check_config() -> CheckResult:
    try:
        config = load_trading_config()
    except Exception as exc:
        return CheckResult("production_config", "fail", str(exc))
    if config.mode != "paper" or config.broker != "paper":
        return CheckResult(
            "production_config",
            "fail",
            "No-VPS staging should keep TRADING_MODE=paper and BROKER=paper",
        )
    if config.enable_live_trading:
        return CheckResult(
            "production_config",
            "fail",
            "ENABLE_LIVE_TRADING must stay false until broker practice/live testing",
        )
    return CheckResult("production_config", "pass", "Safe paper-trading defaults are active")


def _check_market_keys() -> CheckResult:
    expected = ["ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "NEWS_API_KEY"]
    present = _env_present(expected)
    if len(present) == len(expected):
        return CheckResult("market_data_keys", "pass", "Market/news API keys are present")
    missing = sorted(set(expected) - set(present))
    return CheckResult("market_data_keys", "fail", f"Missing API keys: {', '.join(missing)}")


def _check_models() -> CheckResult:
    required = [
        "models/improved_hybrid_trading_agent.zip",
        "models/XAUUSD_model.pkl",
        "models/GBPUSD_model.pkl",
        "models/USDJPY_model.pkl",
        "models/AUDUSD_model.pkl",
    ]
    missing = [path for path in required if not _exists(path)]
    if missing:
        return CheckResult("model_artifacts", "fail", f"Missing model artifacts: {', '.join(missing)}")
    return CheckResult("model_artifacts", "pass", "Production DRL and per-pair ML artifacts exist")


def _check_data() -> CheckResult:
    required = ["XAUUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    missing = []
    thin = []
    for pair in required:
        path = Path("historical_data") / f"{pair}_hourly.csv"
        rows = _count_rows(path)
        if rows == 0:
            missing.append(str(path))
        elif rows < 500:
            thin.append(f"{path} ({rows} rows)")
    if missing:
        return CheckResult("historical_data", "fail", f"Missing or empty data files: {', '.join(missing)}")
    if thin:
        return CheckResult("historical_data", "fail", f"Data files have too few rows: {', '.join(thin)}")
    return CheckResult("historical_data", "pass", "Hourly data files are present with usable history")


def _check_workflows() -> CheckResult:
    required = [
        ".github/workflows/production-trading.yml",
        ".github/workflows/model-retraining.yml",
        ".github/workflows/preflight.yml",
    ]
    missing = [path for path in required if not _exists(path)]
    if missing:
        return CheckResult("github_actions", "fail", f"Missing workflows: {', '.join(missing)}")
    return CheckResult("github_actions", "pass", "Paper check, retraining, and preflight workflows are present")


def _check_deploy_assets() -> CheckResult:
    required = [
        "deploy/trading-bot.service",
        "deploy/trading-bot.timer",
    ]
    missing = [path for path in required if not _exists(path)]
    if missing:
        return CheckResult("deploy_assets", "fail", f"Missing deploy assets: {', '.join(missing)}")
    return CheckResult("deploy_assets", "pass", "Future VPS service files are present")


def _check_broker_deferred() -> CheckResult:
    if os.getenv("OANDA_API_KEY") and os.getenv("OANDA_ACCOUNT_ID"):
        return CheckResult(
            "broker_integration",
            "deferred",
            "Broker credentials exist, but live/practice execution is intentionally deferred",
        )
    return CheckResult(
        "broker_integration",
        "deferred",
        "No broker is required for this stage; integrate practice/live broker after VPS decision",
    )


def _check_vps_deferred() -> CheckResult:
    return CheckResult(
        "vps_hosting",
        "deferred",
        "VPS hosting is intentionally left as the final infrastructure step",
    )


CHECKS: List[Callable[[], CheckResult]] = [
    _check_python_modules,
    _check_cpu_torch,
    _check_config,
    _check_market_keys,
    _check_models,
    _check_data,
    _check_workflows,
    _check_deploy_assets,
    _check_broker_deferred,
    _check_vps_deferred,
]


def run_preflight() -> dict:
    results = [check() for check in CHECKS]
    blocking = [result for result in results if result.status == "fail"]
    return {
        "ready_for_no_vps_stage": not blocking,
        "stage": "local_paper_ready_vps_and_broker_deferred",
        "summary": {
            "pass": sum(1 for result in results if result.status == "pass"),
            "deferred": sum(1 for result in results if result.status == "deferred"),
            "fail": len(blocking),
        },
        "checks": [result.__dict__ for result in results],
        "remaining_before_true_production": [
            "Choose and provision a VPS",
            "Install the repo and .env on the VPS",
            "Choose a broker and validate practice trading",
            "Enable live trading only after paper/practice review",
        ],
    }


def print_text_report(report: dict) -> None:
    status = "READY" if report["ready_for_no_vps_stage"] else "NOT READY"
    print(f"Production staging preflight: {status}")
    print(f"Stage: {report['stage']}")
    print("")
    for check in report["checks"]:
        label = check["status"].upper()
        print(f"[{label}] {check['name']}: {check['message']}")
    print("")
    print("Remaining before true production:")
    for item in report["remaining_before_true_production"]:
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check readiness for no-VPS production staging")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = run_preflight()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    if not report["ready_for_no_vps_stage"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
