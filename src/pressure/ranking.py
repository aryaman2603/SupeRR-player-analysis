"""Ranking, uncertainty and sensitivity.

Produces the final top-10 tables. Bootstrap intervals come from resampling
matches (not balls), so uncertainty reflects whole-match variation rather than
within-innings noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import DEFAULT_PARAMS, OUTPUT_DIR, SENSITIVITY
from .pressure_index import add_pressure_index
from .scoring import score_players, _z

PARAMS = DEFAULT_PARAMS


class ParamsLike:
    """Shallow copy of Params with overrides (keeps the dataclass immutable)."""

    def __init__(self, base, **kwargs):
        self.__dict__.update(base.__dict__)
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _matrix(pm: pd.DataFrame, players: list, matches: list):
    p_idx = {p: i for i, p in enumerate(players)}
    m_idx = {m: i for i, m in enumerate(matches)}
    P, M = len(players), len(matches)
    mats = {k: np.zeros((P, M)) for k in
            ("w", "wr", "ww", "n", "runs_above_sum", "wkts_above_sum",
             "n_all", "runs_above_all", "wkts_above_all")}
    for row in pm.itertuples(index=False):
        i = p_idx.get(row.player_id)
        j = m_idx.get(row.match_id)
        if i is None or j is None:
            continue
        mats["w"][i, j] += row.w
        mats["wr"][i, j] += row.wr
        mats["ww"][i, j] += row.ww
        mats["n"][i, j] += row.n
        mats["runs_above_sum"][i, j] += row.runs_above_sum
        mats["wkts_above_sum"][i, j] += row.wkts_above_sum
        mats["n_all"][i, j] += row.n_all
        mats["runs_above_all"][i, j] += row.runs_above_all
        mats["wkts_above_all"][i, j] += row.wkts_above_all
    return mats


def _composite(prae: np.ndarray, pwae: np.ndarray, params) -> np.ndarray:
    zr = (prae - np.nanmedian(prae)) / (1.4826 * np.nanmedian(np.abs(prae - np.nanmedian(prae))) + 1e-12)
    zw = (pwae - np.nanmedian(pwae)) / (1.4826 * np.nanmedian(np.abs(pwae - np.nanmedian(pwae))) + 1e-12)
    return params.runs_weight * zr + params.wickets_weight * zw


def bootstrap_ranks(pm: pd.DataFrame, scores: pd.DataFrame, params=DEFAULT_PARAMS,
                    reps: int | None = None, seed: int | None = None) -> pd.DataFrame:
    reps = reps or params.bootstrap_reps
    rng = np.random.default_rng(params.seed if seed is None else seed)

    eligible = scores[scores["eligible"]].copy()
    players = eligible["player_id"].tolist()
    matches = sorted(pm["match_id"].unique().tolist())
    mats = _matrix(pm, players, matches)

    lg_runs = float(pm["runs_above_all"].sum() / pm["n_all"].sum())
    lg_wkts = float(pm["wkts_above_all"].sum() / pm["n_all"].sum())
    k0, k = 100.0, params.shrinkage_k
    M = len(matches)
    ranks = np.zeros((reps, len(players)))

    for r in range(reps):
        samp = rng.integers(0, M, size=M)
        counts = np.bincount(samp, minlength=M).astype(float)
        w = mats["w"] @ counts
        wr = mats["wr"] @ counts
        ww = mats["ww"] @ counts
        n_all = mats["n_all"] @ counts
        ra_all = mats["runs_above_all"] @ counts
        wa_all = mats["wkts_above_all"] @ counts

        own_r = (ra_all + k0 * lg_runs) / (n_all + k0)
        own_w = (wa_all + k0 * lg_wkts) / (n_all + k0)
        prae = (wr + k * own_r) / (w + k)
        pwae = (ww + k * own_w) / (w + k)
        comp = _composite(prae, pwae, params)

        order = np.argsort(-comp)
        rep_rank = np.empty(len(players))
        rep_rank[order] = np.arange(1, len(players) + 1)
        ranks[r] = rep_rank

    q_lo, q_hi = (1 - params.bootstrap_ci) / 2, 1 - (1 - params.bootstrap_ci) / 2
    point = eligible.set_index("player_id")["score_index"].rank(ascending=False, method="min")
    return pd.DataFrame(
        {
            "player_id": players,
            "rank_point": point.reindex(players).astype(int).to_numpy(),
            "rank_lo": np.percentile(ranks, 100 * q_lo, axis=0).round(0),
            "rank_median": np.percentile(ranks, 50, axis=0).round(0),
            "rank_hi": np.percentile(ranks, 100 * q_hi, axis=0).round(0),
            "top10_prob": (ranks <= 10).mean(axis=0),
        }
    )


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def _rank_vector(scores: pd.DataFrame) -> pd.Series:
    e = scores[scores["eligible"]]
    return e.set_index("player_id")["composite"]


def _compare(base: pd.Series, alt: pd.Series) -> tuple[float, int, int]:
    common = base.index.intersection(alt.index)
    rho = float(spearmanr(base.loc[common], alt.loc[common]).statistic) if len(common) > 3 else np.nan
    base_top = set(base.sort_values(ascending=False).head(10).index)
    alt_top = set(alt.sort_values(ascending=False).head(10).index)
    return rho, len(base_top & alt_top), len(common)


def run_sensitivity(balls: pd.DataFrame, base_bat: pd.DataFrame, base_bowl: pd.DataFrame,
                    params=DEFAULT_PARAMS) -> pd.DataFrame:
    rows = []
    base_bat_v = _rank_vector(base_bat)
    base_bowl_v = _rank_vector(base_bowl)

    def add_variant(name, bat, bowl):
        rb, ob, nb = _compare(base_bat_v, _rank_vector(bat))
        rw, ow, nw = _compare(base_bowl_v, _rank_vector(bowl))
        rows.append(
            {
                "variant": name,
                "bat_spearman": round(rb, 3) if np.isfinite(rb) else np.nan,
                "bat_top10_overlap": ob, "bat_n": nb,
                "bowl_spearman": round(rw, 3) if np.isfinite(rw) else np.nan,
                "bowl_top10_overlap": ow, "bowl_n": nw,
            }
        )

    def rescore(p):
        b, w, _, _ = score_players(balls, p, save=False)
        return b, w

    for g in SENSITIVITY["gamma"]:
        add_variant(f"gamma={g}", *rescore(ParamsLike(params, gamma=g)))

    for k in SENSITIVITY["shrinkage_k"]:
        add_variant(f"shrinkage_k={k:.0f}", *rescore(ParamsLike(params, shrinkage_k=k)))

    for wt in SENSITIVITY["t20i_weight"]:
        add_variant(
            f"t20i_weight={wt}",
            *rescore(ParamsLike(params, format_weights={"IPL": 1.0, "T20I": wt})),
        )

    for ww in SENSITIVITY["wicket_weight"]:
        add_variant(
            f"wicket_weight={ww}",
            *rescore(ParamsLike(params, runs_weight=1 - ww, wickets_weight=ww)),
        )

    for cut in SENSITIVITY["pressure_cut"]:
        add_variant(
            f"pressure_cut={cut:.0f}",
            *rescore(ParamsLike(params, pressure_sample_cut=cut)),
        )

    for name, changes in {
        "pi_leverage_heavy": {"w_leverage": 0.45, "w_chase": 0.17, "w_wickets": 0.13, "w_phase": 0.13, "w_personal": 0.12},
        "pi_situation_heavy": {"w_leverage": 0.25, "w_chase": 0.25, "w_wickets": 0.20, "w_phase": 0.15, "w_personal": 0.15},
    }.items():
        p = ParamsLike(params, **changes)
        b2 = add_pressure_index(balls, p)
        bb, ww = _score_on(b2, p)
        add_variant(name, bb, ww)

    return pd.DataFrame(rows)


def _score_on(balls: pd.DataFrame, p):
    b, w, _, _ = score_players(balls, p, save=False)
    return b, w


# ---------------------------------------------------------------------------
# Final tables
# ---------------------------------------------------------------------------

def _decorate(table: pd.DataFrame, boot: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = table.merge(boot, on="player_id", how="left", suffixes=("", "_boot"))
    out = out[out["eligible"]].sort_values("score_index", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    keep = ["rank", "player_name", "score_index"] + cols + [
        "rank_lo", "rank_hi", "top10_prob", "clutch_premium",
        "impact_raw_pct", "wkts_above_pct", "share_high",
    ]
    return out[[c for c in keep if c in out.columns]]


def build(params=DEFAULT_PARAMS):
    balls = pd.read_parquet(OUTPUT_DIR / "pressure_balls.parquet")
    bat, bowl, bat_pm, bowl_pm = score_players(balls, params, save=True)

    boot_bat = bootstrap_ranks(bat_pm, bat, params)
    boot_bowl = bootstrap_ranks(bowl_pm, bowl, params)

    bat_cols = [
        "innings", "balls", "n_pressure", "n_eff", "n_high", "pressure_balls",
        "pressure_runs", "pressure_sr", "pressure_avg", "sr_delta",
        "pressure_boundary_pct",
    ]
    bowl_cols = [
        "innings", "balls", "n_pressure", "n_eff", "n_high", "pressure_balls_bowled",
        "pressure_wickets", "pressure_economy", "pressure_bowl_sr", "econ_delta",
        "pressure_dot_pct_bowl",
    ]

    top_bat = _decorate(bat, boot_bat, bat_cols)
    top_bowl = _decorate(bowl, boot_bowl, bowl_cols)

    top_bat.to_csv(OUTPUT_DIR / "ranked_batters.csv", index=False)
    top_bowl.to_csv(OUTPUT_DIR / "ranked_bowlers.csv", index=False)
    top_bat.head(10).to_csv(OUTPUT_DIR / "top10_batters.csv", index=False)
    top_bowl.head(10).to_csv(OUTPUT_DIR / "top10_bowlers.csv", index=False)
    pd.concat([bat, bowl], ignore_index=True).to_csv(
        OUTPUT_DIR / "player_pressure_scores.csv", index=False
    )

    sens = run_sensitivity(balls, bat, bowl, params)
    sens.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)
    return top_bat, top_bowl, sens


if __name__ == "__main__":
    top_bat, top_bowl, sens = build()
    print("=== TOP 10 BATTERS")
    print(top_bat.head(10)[["rank", "player_name", "score_index", "rank_lo", "rank_hi", "top10_prob"]].to_string(index=False))
    print("\n=== TOP 10 BOWLERS")
    print(top_bowl.head(10)[["rank", "player_name", "score_index", "rank_lo", "rank_hi", "top10_prob"]].to_string(index=False))
    print("\n=== SENSITIVITY")
    print(sens.to_string(index=False))
