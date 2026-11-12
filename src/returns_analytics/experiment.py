"""Fixed-seed synthetic price path and reproducible summary chart."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import risk_summary, simple_returns


def run(output_dir: Path, seed: int = 2025) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, size=252)
    prices = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    stats = risk_summary(simple_returns(prices))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(prices.size), prices, color="#0f766e", lw=1.4)
    ax.set(title="Synthetic daily price path (seed 2025)", xlabel="Trading-day index", ylabel="Synthetic price")
    ax.text(0.01, 0.98, "SYNTHETIC • illustrative, not investment performance",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_price_path.png", dpi=150)
    plt.close(fig)
    stats["seed"] = seed
    stats["ending_synthetic_price"] = float(prices[-1])
    (output_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic return report to {args.output_dir}")


if __name__ == "__main__":
    main()

