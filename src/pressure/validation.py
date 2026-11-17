"""Independent checks on the player scores.

1. Repeatability  -> split-half correlation across random match splits.
2. External check -> alignment with Player-of-the-Match awards.
3. Construct check-> correlation with conventional pressure stats.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import DEFAULT_PARAMS, OUTPUT_DIR
from .scoring import add_expectations, _weights, _shrink, _z

PARAMS = DEFAULT_PARAMS


def _prepare(balls: pd.DataFrame, role: str, params=DEFAULT_PARAMS) -> pd.DataFrame:
    b = balls[balls["wp_usable"] & ~balls["is_super_over"]].copy()
    if role == "batter":
        d = b[b["is_legal"] & b["striker_id"].notna() & ~b["nonstriker_runout"]].copy()
        d["player_id"] = d["striker_id"]
        d = add_expectations(d, "bat_runs", "striker_out")
        d["runs_above"] = d["bat_runs"] - d["exp_value"]
        d["wkts_above"] = d["exp_wicket"] - d["striker_out"].astype(float)
    else:
        d = b[b["bowler_id"].notna() & ~b["non_bowler_wicket"]].copy()
        d["player_id"] = d["bowler_id"]
        d = add_expectations(d, "ball_runs", "bowler_credited_wicket")
        d["runs_above"] = d["exp_value"] - d["ball_runs"]
        d["wkts_above"] = d["bowler_credited_wicket"].astype(float) - d["exp_wicket"]
    d["fw"] = d["format"].map(params.format_weights).fillna(0.6)
    return d


def _aggregate(d: pd.DataFrame, params=DEFAULT_PARAMS, min_pressure: int = 75) -> pd.DataFrame:
    h = d[d["pressure_index"] >= params.pressure_sample_cut].copy()
    h["w"] = (h["pressure_index"] / 50.0) ** params.gamma * h["fw"]
    h["wr"] = h["w"] * h["runs_above"]
    h["ww"] = h["w"] * h["wkts_above"]
    a = h.groupby("player_id").agg(
        w=("w", "sum"), wr=("wr", "sum"), ww=("ww", "sum"), np_=("runs_above", "size")
    )
    o = d.groupby("player_id").agg(
        n=("runs_above", "size"), ras=("runs_above", "sum"), was=("wkts_above", "sum")
    )
    a = a.join(o)
    a = a[a["np_"] >= min_pressure]

    lg_r = float(d["runs_above"].mean())
    lg_w = float(d["wkts_above"].mean())
    own_r = (a["ras"] + 100.0 * lg_r) / (a["n"] + 100.0)
    own_w = (a["was"] + 100.0 * lg_w) / (a["n"] + 100.0)
    a["prae"] = _shrink((a["wr"] / a["w"]).to_numpy(), a["w"].to_numpy(), own_r.to_numpy(), params.shrinkage_k)
    a["pwae"] = _shrink((a["ww"] / a["w"]).to_numpy(), a["w"].to_numpy(), own_w.to_numpy(), params.shrinkage_k)
    a["composite"] = params.runs_weight * _z(a["prae"]) + params.wickets_weight * _z(a["pwae"])
    return a


def split_half_reliability(balls: pd.DataFrame, params=DEFAULT_PARAMS, reps: int = 20) -> dict:
    rng = np.random.default_rng(params.seed)
    matches = balls["match_id"].unique()
    out = {}
    for role in ("batter", "bowler"):
        d = _prepare(balls, role, params)
        rhos, n_players = [], 0
        for _ in range(reps):
            sel = set(rng.choice(matches, len(matches) // 2, replace=False))
            a = _aggregate(d[d["match_id"].isin(sel)], params)
            c = _aggregate(d[~d["match_id"].isin(sel)], params)
            common = a.index.intersection(c.index)
            if len(common) > 5:
                rhos.append(spearmanr(a.loc[common, "composite"], c.loc[common, "composite"]).statistic)
                n_players = len(common)
        out[role] = {
            "spearman": round(float(np.nanmean(rhos)), 3) if rhos else np.nan,
            "reps": len(rhos),
            "players_per_split": int(n_players),
        }
    return out


def mom_alignment(balls: pd.DataFrame, params=DEFAULT_PARAMS) -> list[dict]:
    matches = pd.read_csv(OUTPUT_DIR.parent / "data" / "matches.csv")[["match_id", "man_of_the_match"]]
    matches = matches[matches["man_of_the_match"].fillna(0) > 0]
    mom = matches.set_index("match_id")["man_of_the_match"]
    rows = []
    for role in ("batter", "bowler"):
        d = _prepare(balls, role, params)
        d = d[d["pressure_index"] >= params.pressure_sample_cut]
        agg = d.groupby(["match_id", "player_id"]).agg(
            impact=("runs_above", "mean"), balls=("runs_above", "size")
        ).reset_index()
        agg = agg[agg["balls"] >= 6]
        agg["is_mom"] = agg.apply(
            lambda r: float(r["player_id"]) == float(mom.get(r["match_id"], -1)), axis=1
        )
        agg["pct"] = agg.groupby("match_id")["impact"].rank(pct=True)
        sub = agg[agg["is_mom"]]
        rows.append(
            {
                "role": role,
                "matches": int(sub["match_id"].nunique()),
                "mean_mom_percentile": round(float(sub["pct"].mean()), 3),
                "share_mom_top_quartile": round(float((sub["pct"] >= 0.75).mean()), 3),
            }
        )
    return rows


def conventional_agreement(bat: pd.DataFrame, bowl: pd.DataFrame) -> list[dict]:
    out = []
    for role, df, alt in [("batter", bat, "sr_delta"), ("bowler", bowl, "econ_delta")]:
        e = df[df["eligible"]]
        prim = e["prae"]
        secondary = -e[alt] if role == "bowler" else e[alt]
        ok = prim.notna() & secondary.notna()
        rho = float(spearmanr(prim[ok], secondary[ok]).statistic) if ok.sum() > 5 else np.nan
        out.append({"role": role, "spearman_vs_conventional": round(rho, 3), "n": int(ok.sum())})
    return out


def run_validation(balls: pd.DataFrame, bat: pd.DataFrame, bowl: pd.DataFrame) -> dict:
    report = {
        "split_half": split_half_reliability(balls),
        "mom": mom_alignment(balls),
        "conventional": conventional_agreement(bat, bowl),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "validation.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    balls = pd.read_parquet(OUTPUT_DIR / "pressure_balls.parquet")
    bat = pd.read_csv(OUTPUT_DIR / "batter_scores.csv")
    bowl = pd.read_csv(OUTPUT_DIR / "bowler_scores.csv")
    print(json.dumps(run_validation(balls, bat, bowl), indent=2))
