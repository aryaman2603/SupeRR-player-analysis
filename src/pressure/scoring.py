"""Player scoring inside the pressure sample.

Why a pressure sample
---------------------
A continuous weight over every delivery let calm-innings output dominate:
openers and top-order anchors accumulate far more low-pressure balls than
finishers accumulate high-pressure ones.  The score is therefore computed on
the **top ~20% of deliveries by Pressure Index** (PI >= 40).  Within that
sample, more intense pressure still counts more, via (PI/50)^gamma.

Currency
--------
For every pressure delivery we estimate what an average player would have
produced in that exact situation, then measure the residual:

    batter:  runs_above = runs_off_bat - expected_runs
             wkts_above = expected_dismissal_prob - dismissed
    bowler:  runs_above = expected_runs_conceded - runs_conceded
             wkts_above = wicket_taken - expected_wicket_prob

Runs above expectation are about twice as repeatable across random match
splits as win-probability swings, so they carry 80% of the composite. Wickets
carry 20%: rarer, noisier, but still decisive in T20.

Scores are shrunk toward each player's own all-ball level (itself shrunk
toward the league mean) in proportion to how little pressure exposure they
have, then mapped to a 0-100 index with the eligible-player median at 50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, OUTPUT_DIR

PARAMS = DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Situational expectations
# ---------------------------------------------------------------------------

def _situation_frame(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["_chase"] = d["innings"].isin([2, 4]).astype(int)
    d["_wb"] = (d["wkts_before"] // 2).clip(0, 4).astype(int)
    d["_ob"] = (d["overid"] // 4).clip(0, 5).astype(int)
    d["_rb"] = np.where(
        d["_chase"].astype(bool),
        np.floor(d["rrr"].clip(0, 20).fillna(0)),
        -1,
    ).astype(int)
    return d


def add_expectations(d: pd.DataFrame, value_col: str, wicket_col: str,
                     min_bucket: int = 120) -> pd.DataFrame:
    """Attach expected runs and expected wicket probability for each delivery.

    Hierarchical bucket means (finest first, then progressively coarser) keep
    every state covered without a fragile high-dimensional model.
    """
    d = _situation_frame(d)
    keysets = [
        ["_chase", "phase", "_wb", "_ob", "_rb", "format"],
        ["_chase", "phase", "_wb", "format"],
        ["_chase", "phase", "format"],
    ]
    global_v = float(d[value_col].mean())
    global_w = float(d[wicket_col].mean())

    exp_v = pd.Series(np.nan, index=d.index, dtype=float)
    exp_w = pd.Series(np.nan, index=d.index, dtype=float)

    for key in keysets:
        g = d.groupby(key, observed=True)
        mv = g[value_col].transform("mean")
        mw = g[wicket_col].transform("mean")
        n = g[value_col].transform("size")
        use = exp_v.isna() & (n >= min_bucket)
        exp_v.loc[use] = mv.loc[use]
        exp_w.loc[use] = mw.loc[use]

    d["exp_value"] = exp_v.fillna(global_v)
    d["exp_wicket"] = exp_w.fillna(global_w)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(pi: pd.Series, fmt: pd.Series, params=DEFAULT_PARAMS) -> np.ndarray:
    base = (pi.clip(lower=0.0, upper=100.0) / 50.0) ** params.gamma
    fw = fmt.map(params.format_weights).fillna(0.6).to_numpy()
    return base.to_numpy() * fw


def _shrink(observed: np.ndarray, n_eff: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    return (n_eff * observed + k * prior) / (n_eff + k)


def _z(x: pd.Series) -> np.ndarray:
    v = x.to_numpy(dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.nanstd(v) if np.nanstd(v) > 1e-12 else 1.0
    return (v - med) / scale


def _index_from_composite(comp: pd.Series, params=DEFAULT_PARAMS) -> pd.Series:
    z = _z(comp)
    out = params.index_center + params.index_scale * z
    return pd.Series(np.clip(out, 0, 100), index=comp.index)


# ---------------------------------------------------------------------------
# Conventional pressure stats (evidence, not the ranking currency)
# ---------------------------------------------------------------------------

def _pressure_stats_bat(d: pd.DataFrame, cut: float) -> dict:
    p = d[d["pressure_index"] >= cut]
    balls = int(p["is_legal"].sum())
    runs = int(p["bat_runs"].sum())
    outs = int(p["striker_out"].sum())
    all_balls = int(d["is_legal"].sum())
    all_runs = int(d["bat_runs"].sum())
    all_outs = int(d["striker_out"].sum())
    sr = 100.0 * runs / balls if balls else np.nan
    sr_all = 100.0 * all_runs / all_balls if all_balls else np.nan
    avg = runs / outs if outs else np.nan
    avg_all = all_runs / all_outs if all_outs else np.nan
    return {
        "pressure_balls": balls,
        "pressure_runs": runs,
        "pressure_sr": round(sr, 1) if np.isfinite(sr) else np.nan,
        "pressure_outs": outs,
        "pressure_avg": round(avg, 1) if np.isfinite(avg) else np.nan,
        "pressure_boundary_pct": round(100.0 * (p["is_four"].sum() + p["is_six"].sum()) / balls, 1) if balls else np.nan,
        "pressure_dot_pct": round(100.0 * ((p["bat_runs"] == 0) & p["is_legal"]).sum() / balls, 1) if balls else np.nan,
        "bat_sr_all": round(sr_all, 1) if np.isfinite(sr_all) else np.nan,
        "sr_delta": round(sr - sr_all, 1) if np.isfinite(sr) and np.isfinite(sr_all) else np.nan,
        "bat_avg_all": round(avg_all, 1) if np.isfinite(avg_all) else np.nan,
        "avg_delta": round(avg - avg_all, 1) if np.isfinite(avg) and np.isfinite(avg_all) else np.nan,
    }


def _pressure_stats_bowl(d: pd.DataFrame, cut: float) -> dict:
    p = d[d["pressure_index"] >= cut]
    balls = int(p["is_legal"].sum())
    runs = int(p["ball_runs"].sum())
    wkts = int(p["bowler_credited_wicket"].sum())
    all_balls = int(d["is_legal"].sum())
    all_runs = int(d["ball_runs"].sum())
    econ = 6.0 * runs / balls if balls else np.nan
    econ_all = 6.0 * all_runs / all_balls if all_balls else np.nan
    return {
        "pressure_balls_bowled": balls,
        "pressure_runs_conceded": runs,
        "pressure_economy": round(econ, 2) if np.isfinite(econ) else np.nan,
        "pressure_wickets": wkts,
        "pressure_bowl_avg": round(runs / wkts, 1) if wkts else np.nan,
        "pressure_bowl_sr": round(balls / wkts, 1) if wkts else np.nan,
        "pressure_dot_pct_bowl": round(100.0 * ((p["ball_runs"] == 0) & p["is_legal"]).sum() / balls, 1) if balls else np.nan,
        "bowl_econ_all": round(econ_all, 2) if np.isfinite(econ_all) else np.nan,
        "econ_delta": round(econ - econ_all, 2) if np.isfinite(econ) and np.isfinite(econ_all) else np.nan,
    }


# ---------------------------------------------------------------------------
# Batter scores
# ---------------------------------------------------------------------------

def batter_scores(balls: pd.DataFrame, params=DEFAULT_PARAMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_all = balls[
        balls["wp_usable"]
        & balls["is_legal"]
        & balls["striker_id"].notna()
        & (~balls["nonstriker_runout"])
    ].copy()
    d_all = add_expectations(d_all, "bat_runs", "striker_out")
    d_all["runs_above"] = d_all["bat_runs"] - d_all["exp_value"]
    d_all["wkts_above"] = d_all["exp_wicket"] - d_all["striker_out"].astype(float)

    # Pressure sample.
    d = d_all[d_all["pressure_index"] >= params.pressure_sample_cut].copy()
    d["w"] = _weights(d["pressure_index"], d["format"], params)
    d["wr"] = d["w"] * d["runs_above"]
    d["ww"] = d["w"] * d["wkts_above"]

    pm = (
        d.groupby(["striker_id", "match_id"])
        .agg(
            w=("w", "sum"), wr=("wr", "sum"), ww=("ww", "sum"),
            n=("runs_above", "size"),
            runs_above_sum=("runs_above", "sum"),
            wkts_above_sum=("wkts_above", "sum"),
            runs=("bat_runs", "sum"), legal=("is_legal", "sum"), outs=("striker_out", "sum"),
        )
        .reset_index()
        .rename(columns={"striker_id": "player_id"})
    )
    pm_all = (
        d_all.groupby(["striker_id", "match_id"])
        .agg(
            n_all=("runs_above", "size"),
            runs_above_all=("runs_above", "sum"),
            wkts_above_all=("wkts_above", "sum"),
        )
        .reset_index()
        .rename(columns={"striker_id": "player_id"})
    )
    pm = pm.merge(pm_all, on=["player_id", "match_id"], how="left")

    g = d.groupby("striker_id", sort=False)
    out = g.apply(
        lambda x: pd.Series(
            {
                "player_name": x["striker_name"].mode().iloc[0],
                "matches": x["match_id"].nunique(),
                "pressure_balls_sample": len(x),
                "w_sum": x["w"].sum(),
                "wr_sum": x["wr"].sum(),
                "ww_sum": x["ww"].sum(),
                "runs_above_sum": x["runs_above"].sum(),
                "wkts_above_sum": x["wkts_above"].sum(),
                "pi_mean": x["pressure_index"].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()

    # All-ball context: innings, total balls/runs, and shrinkage priors.
    overall = d_all.groupby("striker_id", sort=False).agg(
        matches_all=("match_id", "nunique"),
        balls=("is_legal", "sum"),
        runs=("bat_runs", "sum"),
        outs=("striker_out", "sum"),
        n_all=("runs_above", "size"),
        runs_above_all=("runs_above", "sum"),
        wkts_above_all=("wkts_above", "sum"),
    ).reset_index()
    innings = d_all.groupby("striker_id").apply(
        lambda x: x[["match_id", "innings"]].drop_duplicates().shape[0], include_groups=False
    ).rename("innings")
    overall = overall.merge(innings.reset_index(), on="striker_id", how="left")
    out = out.merge(overall, on="striker_id", how="left")

    out["prae_raw"] = out["wr_sum"] / out["w_sum"]
    out["pwae_raw"] = out["ww_sum"] / out["w_sum"]
    out["n_eff"] = out["w_sum"]
    out["n_pressure"] = out["pressure_balls_sample"]

    # Shrinkage prior: the player's own all-ball level, itself shrunk to league.
    lg_runs = float(d_all["runs_above"].mean())
    lg_wkts = float(d_all["wkts_above"].mean())
    k0 = 100.0
    out["own_runs"] = (out["runs_above_all"] + k0 * lg_runs) / (out["n_all"] + k0)
    out["own_wkts"] = (out["wkts_above_all"] + k0 * lg_wkts) / (out["n_all"] + k0)
    out["prae"] = _shrink(out["prae_raw"].to_numpy(), out["n_eff"].to_numpy(), out["own_runs"].to_numpy(), params.shrinkage_k)
    out["pwae"] = _shrink(out["pwae_raw"].to_numpy(), out["n_eff"].to_numpy(), out["own_wkts"].to_numpy(), params.shrinkage_k)

    # Phase sub-scores, on the pressure sample: runs above expectation / 100 balls.
    for phase in ("powerplay", "middle", "death"):
        ph = d[d["phase"] == phase]
        agg = ph.groupby("striker_id").apply(
            lambda x: pd.Series({"w": x["w"].sum(), "wr": x["wr"].sum()}), include_groups=False
        )
        agg["v"] = np.where(agg["w"] > 0, agg["wr"] / agg["w"].replace(0, np.nan), np.nan)
        out[f"phase_{phase}"] = out["striker_id"].map(agg["v"])
        n_ph = out["striker_id"].map(agg["w"]).fillna(0.0)
        out[f"phase_{phase}"] = _shrink(
            out[f"phase_{phase}"].fillna(out["own_runs"]).to_numpy(),
            n_ph.to_numpy(), out["own_runs"].to_numpy(), params.shrinkage_k / 2.0,
        ) * 100.0

    # Clutch premium: situation-adjusted runs per ball, high minus calm.
    high = d_all[d_all["pressure_index"] >= params.pressure_cut].groupby("striker_id")["runs_above"]
    calm = d_all[d_all["pressure_index"] < params.calm_cut].groupby("striker_id")["runs_above"]
    out["clutch_premium"] = (out["striker_id"].map(high.mean()) - out["striker_id"].map(calm.mean())) * 100.0
    out["n_high"] = out["striker_id"].map(high.size()).fillna(0)
    out["n_calm"] = out["striker_id"].map(calm.size()).fillna(0)

    out["impact_raw_pct"] = 100.0 * out["prae_raw"]
    out["impact_high_pct"] = 100.0 * out["striker_id"].map(high.mean())
    out["wkts_above_pct"] = 100.0 * out["pwae_raw"]
    out["share_high"] = out["n_high"] / out["balls"].clip(lower=1)

    stats = [
        {"striker_id": pid, **_pressure_stats_bat(x, params.pressure_cut)}
        for pid, x in d_all.groupby("striker_id", sort=False)
    ]
    out = out.merge(pd.DataFrame(stats), on="striker_id", how="left")
    out["role"] = "batter"
    out = out.rename(columns={"striker_id": "player_id"})
    return out, pm


# ---------------------------------------------------------------------------
# Bowler scores
# ---------------------------------------------------------------------------

def bowler_scores(balls: pd.DataFrame, params=DEFAULT_PARAMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_all = balls[
        balls["wp_usable"]
        & balls["bowler_id"].notna()
        & (~balls["non_bowler_wicket"])
    ].copy()
    d_all = add_expectations(d_all, "ball_runs", "bowler_credited_wicket")
    d_all["runs_above"] = d_all["exp_value"] - d_all["ball_runs"]
    d_all["wkts_above"] = d_all["bowler_credited_wicket"].astype(float) - d_all["exp_wicket"]

    d = d_all[d_all["pressure_index"] >= params.pressure_sample_cut].copy()
    d["w"] = _weights(d["pressure_index"], d["format"], params)
    d["wr"] = d["w"] * d["runs_above"]
    d["ww"] = d["w"] * d["wkts_above"]

    pm = (
        d.groupby(["bowler_id", "match_id"])
        .agg(
            w=("w", "sum"), wr=("wr", "sum"), ww=("ww", "sum"),
            n=("runs_above", "size"),
            runs_above_sum=("runs_above", "sum"),
            wkts_above_sum=("wkts_above", "sum"),
            legal=("is_legal", "sum"), runs=("ball_runs", "sum"),
            wkts=("bowler_credited_wicket", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    pm_all = (
        d_all.groupby(["bowler_id", "match_id"])
        .agg(
            n_all=("runs_above", "size"),
            runs_above_all=("runs_above", "sum"),
            wkts_above_all=("wkts_above", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    pm = pm.merge(pm_all, on=["player_id", "match_id"], how="left")

    g = d.groupby("bowler_id", sort=False)
    out = g.apply(
        lambda x: pd.Series(
            {
                "player_name": x["bowler_name"].mode().iloc[0],
                "matches": x["match_id"].nunique(),
                "pressure_balls_sample": len(x),
                "w_sum": x["w"].sum(),
                "wr_sum": x["wr"].sum(),
                "ww_sum": x["ww"].sum(),
                "runs_above_sum": x["runs_above"].sum(),
                "wkts_above_sum": x["wkts_above"].sum(),
                "pi_mean": x["pressure_index"].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()

    overall = d_all.groupby("bowler_id", sort=False).agg(
        matches_all=("match_id", "nunique"),
        balls=("is_legal", "sum"),
        runs_conceded=("ball_runs", "sum"),
        wickets=("bowler_credited_wicket", "sum"),
        n_all=("runs_above", "size"),
        runs_above_all=("runs_above", "sum"),
        wkts_above_all=("wkts_above", "sum"),
    ).reset_index()
    innings = d_all.groupby("bowler_id").apply(
        lambda x: x[["match_id", "innings"]].drop_duplicates().shape[0], include_groups=False
    ).rename("innings")
    overall = overall.merge(innings.reset_index(), on="bowler_id", how="left")
    out = out.merge(overall, on="bowler_id", how="left")

    out["prae_raw"] = out["wr_sum"] / out["w_sum"]
    out["pwae_raw"] = out["ww_sum"] / out["w_sum"]
    out["n_eff"] = out["w_sum"]
    out["n_pressure"] = out["pressure_balls_sample"]

    lg_runs = float(d_all["runs_above"].mean())
    lg_wkts = float(d_all["wkts_above"].mean())
    k0 = 100.0
    out["own_runs"] = (out["runs_above_all"] + k0 * lg_runs) / (out["n_all"] + k0)
    out["own_wkts"] = (out["wkts_above_all"] + k0 * lg_wkts) / (out["n_all"] + k0)
    out["prae"] = _shrink(out["prae_raw"].to_numpy(), out["n_eff"].to_numpy(), out["own_runs"].to_numpy(), params.shrinkage_k)
    out["pwae"] = _shrink(out["pwae_raw"].to_numpy(), out["n_eff"].to_numpy(), out["own_wkts"].to_numpy(), params.shrinkage_k)

    for phase in ("powerplay", "middle", "death"):
        ph = d[d["phase"] == phase]
        agg = ph.groupby("bowler_id").apply(
            lambda x: pd.Series({"w": x["w"].sum(), "wr": x["wr"].sum()}), include_groups=False
        )
        agg["v"] = np.where(agg["w"] > 0, agg["wr"] / agg["w"].replace(0, np.nan), np.nan)
        out[f"phase_{phase}"] = out["bowler_id"].map(agg["v"])
        n_ph = out["bowler_id"].map(agg["w"]).fillna(0.0)
        out[f"phase_{phase}"] = _shrink(
            out[f"phase_{phase}"].fillna(out["own_runs"]).to_numpy(),
            n_ph.to_numpy(), out["own_runs"].to_numpy(), params.shrinkage_k / 2.0,
        ) * 100.0

    high = d_all[d_all["pressure_index"] >= params.pressure_cut].groupby("bowler_id")["runs_above"]
    calm = d_all[d_all["pressure_index"] < params.calm_cut].groupby("bowler_id")["runs_above"]
    out["clutch_premium"] = (out["bowler_id"].map(high.mean()) - out["bowler_id"].map(calm.mean())) * 100.0
    out["n_high"] = out["bowler_id"].map(high.size()).fillna(0)
    out["n_calm"] = out["bowler_id"].map(calm.size()).fillna(0)

    out["impact_raw_pct"] = 100.0 * out["prae_raw"]
    out["impact_high_pct"] = 100.0 * out["bowler_id"].map(high.mean())
    out["wkts_above_pct"] = 100.0 * out["pwae_raw"]
    out["share_high"] = out["n_high"] / out["balls"].clip(lower=1)

    stats = [
        {"bowler_id": pid, **_pressure_stats_bowl(x, params.pressure_cut)}
        for pid, x in d_all.groupby("bowler_id", sort=False)
    ]
    out = out.merge(pd.DataFrame(stats), on="bowler_id", how="left")
    out["role"] = "bowler"
    out = out.rename(columns={"bowler_id": "player_id"})
    return out, pm


# ---------------------------------------------------------------------------
# Composite scoring and eligibility
# ---------------------------------------------------------------------------

def _finalise(table: pd.DataFrame, params=DEFAULT_PARAMS, min_pressure: int = 150) -> pd.DataFrame:
    table = table.copy()
    table["eligible"] = (
        (table["n_pressure"] >= min_pressure)
        & (table["innings"] >= params.min_innings)
    )
    table["composite"] = np.nan
    table["score_index"] = np.nan
    elig = table["eligible"].to_numpy()
    if elig.sum() >= 3:
        sub = table.loc[elig].copy()
        sub["composite"] = params.runs_weight * _z(sub["prae"]) + params.wickets_weight * _z(sub["pwae"])
        sub["score_index"] = _index_from_composite(sub["composite"], params)
        table.loc[elig, "composite"] = sub["composite"].to_numpy()
        table.loc[elig, "score_index"] = sub["score_index"].to_numpy()
    table["eligible"] = table["eligible"].astype(bool)
    return table


def score_players(balls: pd.DataFrame, params=DEFAULT_PARAMS, save: bool = True):
    bat, bat_pm = batter_scores(balls, params)
    bowl, bowl_pm = bowler_scores(balls, params)

    bat = _finalise(bat, params, params.min_pressure_balls_bat)
    bowl = _finalise(bowl, params, params.min_pressure_balls_bowl)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bat.to_csv(OUTPUT_DIR / "batter_scores.csv", index=False)
        bowl.to_csv(OUTPUT_DIR / "bowler_scores.csv", index=False)
        bat_pm.to_parquet(OUTPUT_DIR / "batter_match_agg.parquet", index=False)
        bowl_pm.to_parquet(OUTPUT_DIR / "bowler_match_agg.parquet", index=False)
    return bat, bowl, bat_pm, bowl_pm


if __name__ == "__main__":
    b = pd.read_parquet(OUTPUT_DIR / "pressure_balls.parquet")
    bat, bowl, _, _ = score_players(b)
    bcols = ["player_name", "innings", "balls", "n_pressure", "n_eff", "score_index", "clutch_premium", "impact_raw_pct", "pressure_sr"]
    wcols = ["player_name", "innings", "balls", "n_pressure", "n_eff", "score_index", "clutch_premium", "impact_raw_pct", "pressure_economy", "pressure_wickets"]
    print("=== eligible batters:", bat["eligible"].sum(), "of", len(bat))
    print(bat[bat.eligible].sort_values("score_index", ascending=False)[bcols].head(15).to_string(index=False))
    print("\n=== eligible bowlers:", bowl["eligible"].sum(), "of", len(bowl))
    print(bowl[bowl.eligible].sort_values("score_index", ascending=False)[wcols].head(15).to_string(index=False))
