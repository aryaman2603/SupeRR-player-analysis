"""End-to-end orchestrator: raw CSVs -> deck-ready outputs.

Usage
-----
    uv run python -m src.pressure.pipeline            # full rebuild
    uv run python -m src.pressure.pipeline --no-figs  # skip charts
    uv run python -m src.pressure.pipeline --from-wp  # reuse ball state

Every step is deterministic (seed 42) and writes into ``outputs/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR
from .figures import build_all as build_figures
from .pressure_index import build_pressure_index
from .ranking import build as build_ranking
from .scoring import score_players
from .validation import run_validation
from .wp_model import compute_win_probability


def _step(name: str, fn, *args, **kwargs):
    t0 = time.time()
    print(f"[pipeline] {name} ...", flush=True)
    result = fn(*args, **kwargs)
    print(f"[pipeline] {name} done in {time.time() - t0:,.1f}s", flush=True)
    return result


def run(skip_data: bool = False, skip_figures: bool = False) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if skip_data and (OUTPUT_DIR / "pressure_balls_state.parquet").exists():
        state = pd.read_parquet(OUTPUT_DIR / "pressure_balls_state.parquet")
        print(f"[pipeline] reusing {len(state):,} ball-state rows")
    else:
        from .data import build_state

        state = _step("data / state reconstruction", build_state)

    with_wp, wp_metrics = _step("win-probability model", compute_win_probability, state)
    balls = _step("pressure index", build_pressure_index, with_wp)
    bat, bowl, _, _ = _step("player scoring", score_players, balls)
    top_bat, top_bowl, sens = _step("rankings / bootstrap", build_ranking)
    validation = _step("validation", run_validation, balls, bat, bowl)

    if not skip_figures:
        _step("figures", build_figures)

    print("[pipeline] complete")
    print(f"  batters eligible: {int(bat['eligible'].sum())}")
    print(f"  bowlers eligible: {int(bowl['eligible'].sum())}")
    print(f"  top batter: {top_bat.iloc[0]['player_name']} ({top_bat.iloc[0]['score_index']:.1f})")
    print(f"  top bowler: {top_bowl.iloc[0]['player_name']} ({top_bowl.iloc[0]['score_index']:.1f})")

    return {
        "balls": balls,
        "bat": bat,
        "bowl": bowl,
        "top_bat": top_bat,
        "top_bowl": top_bowl,
        "sensitivity": sens,
        "validation": validation,
        "wp_metrics": wp_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the pressure-performance analysis.")
    parser.add_argument("--no-figs", action="store_true", help="skip chart generation")
    parser.add_argument("--from-wp", action="store_true",
                        help="reuse the cached ball-state parquet (skip raw CSV processing)")
    args = parser.parse_args()
    run(skip_data=args.from_wp, skip_figures=args.no_figs)


if __name__ == "__main__":
    main()
