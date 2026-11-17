"""Win-probability model.

A calibrated probability that the batting side wins, given the match state.
It powers two things:

1. Leverage for the Pressure Index  ->  4 * p * (1 - p)
2. Ball-level impact (delta WP), the currency players are scored in.

A regularised logistic regression beat gradient boosting in out-of-fold
validation on both innings (the WP surface is smooth, so the simpler model
generalises better). Predictions are cross-fitted by match, so a player is
never scored with a model that has seen that match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_PARAMS, OUTPUT_DIR

PARAMS = DEFAULT_PARAMS
FEATURES = {
    1: [
        "score", "wkts", "legal", "balls_rem", "crr", "projected", "proj_vs_par",
        "score_x_wktsleft", "par_x_wktsleft", "log_proj", "sqrt_balls",
        "elo_diff", "venue_par", "is_ipl",
    ],
    2: [
        "score", "wkts", "legal", "balls_rem", "crr", "runs_needed", "rrr", "demand",
        "rrr_x_wktsleft", "demand_x_balls", "log_runs", "log_balls", "wktsleft",
        "needed_per_wkt", "target", "elo_diff", "venue_par", "is_ipl",
    ],
}


def _design(df: pd.DataFrame, innings: int, prefix: str = "") -> pd.DataFrame:
    """Feature matrix for a live innings state.

    ``prefix`` builds the post-delivery variant (``after``) from the same formula.
    """
    if prefix:
        score = df["runs_after"]
        wkts = df["wkts_after"]
        legal = df["legal_after"]
    else:
        score = df["runs_before"]
        wkts = df["wkts_before"]
        legal = df["legal_before"]

    balls_total = df["innings_overs"] * 6.0
    balls_rem = np.maximum(0.0, balls_total - legal)
    crr = score * 6.0 / np.maximum(1.0, legal)
    proj = score * balls_total / np.maximum(1.0, legal)
    wkts_left = 10.0 - wkts

    out = pd.DataFrame(
        {
            "score": score,
            "wkts": wkts,
            "legal": legal,
            "balls_rem": balls_rem,
            "crr": crr,
            "wktsleft": wkts_left,
            "elo_diff": df["bat_elo"] - df["bowl_elo"],
            "venue_par": df["venue_par"],
            "is_ipl": df["is_ipl"],
        }
    )

    if innings == 1:
        out["projected"] = proj
        out["proj_vs_par"] = proj - df["venue_par"]
        out["score_x_wktsleft"] = score * wkts_left
        out["par_x_wktsleft"] = (proj - df["venue_par"]) * wkts_left
        out["log_proj"] = np.log1p(np.maximum(0.0, proj))
        out["sqrt_balls"] = np.sqrt(balls_rem)
    else:
        runs_needed = df["target"] - score
        out["runs_needed"] = runs_needed
        out["rrr"] = np.clip(runs_needed * 6.0 / np.maximum(1.0, balls_rem), 0.0, 36.0)
        out["demand"] = np.clip(runs_needed / np.maximum(1.0, balls_rem), 0.0, 6.0)
        out["rrr_x_wktsleft"] = out["rrr"] * wkts_left
        out["demand_x_balls"] = out["demand"] * balls_rem
        out["log_runs"] = np.log1p(np.maximum(0.0, runs_needed))
        out["log_balls"] = np.log1p(balls_rem)
        out["needed_per_wkt"] = runs_needed / wkts_left.clip(lower=1.0)
        out["target"] = df["target"]
    return out[FEATURES[innings]]


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000, random_state=PARAMS.seed),
    )


def _fit_predict_oof(
    df: pd.DataFrame, innings: int, label: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cross-fitted before/after win probabilities for one innings type."""
    X_before = _design(df, innings)
    X_after = _design(df, innings, prefix="after")
    p_before = np.full(len(df), np.nan)
    p_after = np.full(len(df), np.nan)

    groups = df["match_id"].to_numpy()
    gkf = GroupKFold(n_splits=PARAMS.n_folds)
    fold_metrics = []
    for tr, te in gkf.split(X_before, label, groups=groups):
        model = _model()
        model.fit(X_before.iloc[tr], label[tr])
        p_before[te] = model.predict_proba(X_before.iloc[te])[:, 1]
        p_after[te] = model.predict_proba(X_after.iloc[te])[:, 1]
        fold_metrics.append(
            {
                "brier": brier_score_loss(label[te], p_before[te]),
                "logloss": log_loss(label[te], np.clip(p_before[te], 1e-6, 1 - 1e-6)),
            }
        )
    metrics = {
        "brier": float(np.mean([f["brier"] for f in fold_metrics])),
        "logloss": float(np.mean([f["logloss"] for f in fold_metrics])),
        "baseline_brier": float(((label - label.mean()) ** 2).mean()),
        "n": int(len(df)),
        "base_rate": float(np.mean(label)),
    }
    return p_before, p_after, metrics


def compute_win_probability(balls: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach ``wp_before``, ``wp_after`` and ``dwp`` to the ball table."""
    b = balls.copy()

    usable = (
        (~b["no_result"].astype(bool))
        & (b["state_consistent"].astype(bool))
        & (~b["is_super_over"].astype(bool))
    )
    b["wp_usable"] = usable

    metrics: dict = {}

    for innings in (1, 2):
        mask = usable & (b["innings"] == innings)
        sub = b.loc[mask].reset_index()
        y = sub["batting_team_won"].astype(float).to_numpy()

        p_before, p_after, met = _fit_predict_oof(sub, innings, y)

        # Terminal ball of a chase: the match is decided right here. Only
        # applies to the second innings -- the first innings break leaves the
        # match live, so the model's post-state probability is correct there.
        if innings == 2:
            balls_total = sub["innings_overs"] * 6.0
            decided = (
                (sub["runs_after"] >= sub["target"])
                | (sub["wkts_after"] >= 10)
                | ((balls_total - sub["legal_after"]) <= 0)
            )
            p_after = np.where(decided, y, p_after)

        b.loc[sub["index"], "wp_before"] = p_before
        b.loc[sub["index"], "wp_after"] = p_after
        metrics[f"innings_{innings}"] = met

    b["dwp"] = b["wp_after"] - b["wp_before"]
    b.loc[~b["wp_usable"], ["wp_before", "wp_after", "dwp"]] = np.nan

    metrics["mean_dwp"] = float(b.loc[b["wp_usable"], "dwp"].mean())
    metrics["dwp_sd"] = float(b.loc[b["wp_usable"], "dwp"].std())

    # Reliability curve (out-of-fold) for the deck.
    live = b[b["wp_usable"]].copy()
    live["bin"] = pd.cut(live["wp_before"], bins=np.linspace(0, 1, 11), include_lowest=True)
    cal = (
        live.groupby("bin", observed=True)
        .agg(predicted=("wp_before", "mean"), observed=("batting_team_won", "mean"), n=("wp_before", "size"))
        .reset_index()
    )
    cal.to_csv(OUTPUT_DIR / "wp_calibration.csv", index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b.to_parquet(OUTPUT_DIR / "pressure_balls_wp.parquet", index=False)
    pd.Series(metrics).to_json(OUTPUT_DIR / "wp_metrics.json", indent=2)
    return b, metrics


if __name__ == "__main__":
    from .data import build_state

    state = build_state()
    out, met = compute_win_probability(state)
    print(pd.Series(met).to_string())
