"""Pressure Index: score every delivery 0-100 on how much pressure it carries.

Five normalised components, multiplied by a knockout/stakes factor:

    PI = 100 * clip( S * (wL*L + wC*C + wW*W + wP*P + wM*M), 0, 1 )

L  leverage        how much win probability this match state typically swings
C  chase/game      required rate outrunning the current rate, or behind venue par
W  wicket scarcity wickets lost / 9
P  phase           powerplay 0.3, middle 0.2, death 1.0
M  personal        nervous 90s, new batter, set batter, final over, hat-trick
S  stakes          league 1.0 -> final 1.3

Leverage is measured from the data rather than assumed: the within-state
standard deviation of delta win probability. A 50/50 score at 2 overs is not
as tense as a 50/50 score at 19, and this captures that automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, OUTPUT_DIR

PARAMS = DEFAULT_PARAMS


def _leverage(df: pd.DataFrame, params=DEFAULT_PARAMS) -> np.ndarray:
    """Within-state swing potential of a delivery, normalised to 0-1."""
    is_chase = df["innings"].isin([2, 4])
    work = pd.DataFrame(
        {
            "dwp": df["dwp"],
            "is_chase": is_chase.astype(int).to_numpy(),
            "phase": df["phase"].to_numpy(),
            "wkts_band": (df["wkts_before"] // 2).clip(0, 4).astype(int).to_numpy(),
            "over_band": (df["overid"] // 4).clip(0, 5).astype(int).to_numpy(),
            "rrr_band": np.where(
                is_chase, np.floor(df["rrr"].clip(0, 20).fillna(0)), -1.0
            ).astype(int),
        },
        index=df.index,
    )
    grp = work.groupby(
        ["is_chase", "phase", "wkts_band", "over_band", "rrr_band"], observed=True
    )["dwp"]
    sd = grp.transform("std")
    n = grp.transform("size")
    global_sd = float(df["dwp"].std())
    # Shrink sparse states toward the global swing scale.
    sd = (sd.fillna(global_sd) * n + global_sd * 30.0) / (n + 30.0)
    ceiling = float(sd.quantile(0.97))
    return np.clip(sd / max(ceiling, 1e-6), 0, 1).to_numpy()


def _chase_component(df: pd.DataFrame) -> np.ndarray:
    """How far behind the game the batting side is."""
    chase = np.clip((df["rrr"].fillna(0) - df["crr"]) / 6.0, 0, 1)
    frac = df["legal_before"] / (df["innings_overs"] * 6.0).clip(lower=1)
    expected = df["venue_par"] * frac
    behind = (expected - df["runs_before"]) / (df["venue_par"] * 0.35).clip(lower=1)
    setting = np.clip(behind, 0, 1)
    return np.where(df["innings"].isin([2, 4]), chase, setting)


def _personal_component(df: pd.DataFrame) -> np.ndarray:
    batter = np.maximum(
        df["batter_nervous_nineties"].astype(float) * PARAMS.p_nervous_nineties,
        df["batter_new"].astype(float) * PARAMS.p_new_batter,
    )
    bowler = np.maximum.reduce(
        [
            df["bowler_final_over"].astype(float) * PARAMS.p_final_over,
            df["bowler_hat_trick"].astype(float) * PARAMS.p_hat_trick,
            df["batter_set"].astype(float) * PARAMS.p_set_batter,
        ]
    )
    return np.clip(np.maximum(batter, bowler), 0, 1)


def _phase_component(df: pd.DataFrame) -> np.ndarray:
    return df["phase"].map({"powerplay": 0.3, "middle": 0.2, "death": 1.0}).fillna(0.2).to_numpy()


def add_pressure_index(balls: pd.DataFrame, params=DEFAULT_PARAMS) -> pd.DataFrame:
    b = balls.copy()

    b["pi_leverage"] = _leverage(b, params)
    b["pi_chase"] = _chase_component(b)
    b["pi_wickets"] = np.clip(b["wkts_before"] / 9.0, 0, 1)
    b["pi_phase"] = _phase_component(b)
    b["pi_personal"] = _personal_component(b)

    raw = (
        params.w_leverage * b["pi_leverage"]
        + params.w_chase * b["pi_chase"]
        + params.w_wickets * b["pi_wickets"]
        + params.w_phase * b["pi_phase"]
        + params.w_personal * b["pi_personal"]
    )
    b["pressure_index"] = 100.0 * np.clip(raw * b["stakes"], 0, 1)
    b.loc[~b["wp_usable"], "pressure_index"] = np.nan
    return b


def build_pressure_index(balls: pd.DataFrame) -> pd.DataFrame:
    b = add_pressure_index(balls)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b.to_parquet(OUTPUT_DIR / "pressure_balls.parquet", index=False)
    return b


if __name__ == "__main__":
    from .data import build_state
    from .wp_model import compute_win_probability

    state = build_state()
    with_wp, _ = compute_win_probability(state)
    b = build_pressure_index(with_wp)
    d = b["pressure_index"].dropna()
    print(d.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print("\nshare PI>=60:", round((d >= 60).mean(), 4), " share PI>=45:", round((d >= 45).mean(), 4))
    print("\nmean PI by phase / innings type:")
    b["inn_type"] = np.where(b["innings"].isin([2, 4]), "chase", "setting")
    print(
        b.groupby(["inn_type", "phase"], observed=True)["pressure_index"]
        .mean().round(1).to_string()
    )
