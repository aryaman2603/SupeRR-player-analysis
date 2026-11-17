"""Load the raw CSVs and reconstruct ball-level match state.

The reconstructed table is the foundation of everything downstream:
pressure needs to know the score, wickets, balls remaining, target and
which batter/bowler was involved on every delivery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, ILLEGAL_DELIVERIES, OUTPUT_DIR, STAGE_STAKES


def load_raw() -> dict[str, pd.DataFrame]:
    """Read the six supplied competition CSVs and keep only IPL / T20I.

    The source files also contain a handful of other T20 competitions
    (e.g. Emerging Teams Asia Cup).  They are out of scope for this
    analysis and are dropped here so every downstream table has the same
    IPL + T20I universe.
    """
    raw = {
        "teams": pd.read_csv(DATA_DIR / "teams.csv"),
        "matches": pd.read_csv(DATA_DIR / "matches.csv"),
        "ball_by_ball": pd.read_csv(DATA_DIR / "ball_by_ball.csv", low_memory=False),
        "competitions": pd.read_csv(DATA_DIR / "competitions.csv"),
        "mapping": pd.read_csv(DATA_DIR / "competition_team_player_mapping.csv"),
        "playing_eleven": pd.read_csv(DATA_DIR / "playing_eleven.csv"),
    }

    comps = raw["competitions"]
    keep_comps = comps[
        comps["match_format"].eq("T20I")
        | comps["comp_name"].str.contains("ipl", case=False, na=False)
    ]["comp_id"]
    matches = raw["matches"][raw["matches"]["comp_id"].isin(keep_comps)].copy()
    keep_matches = set(matches["match_id"])

    raw["matches"] = matches
    raw["ball_by_ball"] = raw["ball_by_ball"][
        raw["ball_by_ball"]["match_id"].isin(keep_matches)
    ].copy()
    raw["playing_eleven"] = raw["playing_eleven"][
        raw["playing_eleven"]["match_id"].isin(keep_matches)
    ].copy()
    raw["mapping"] = raw["mapping"][
        raw["mapping"]["comp_id"].isin(keep_comps)
    ].copy()
    raw["competitions"] = comps[comps["comp_id"].isin(keep_comps)].copy()
    return raw


# ---------------------------------------------------------------------------
# Match-level enrichment: format, stakes, Elo, venue par, winner, totals
# ---------------------------------------------------------------------------

def _format_from_comp(comp_name: str, match_format: str) -> str:
    if "ipl" in str(comp_name).lower():
        return "IPL"
    return "T20I" if str(match_format).upper() == "T20I" else "T20"


def _process_elo(matches: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Chronological Elo rating per team, stored pre-match (leak-free).

    Margin of victory scales the update, following the standard cricket-Elo
    convention. No-result matches leave ratings unchanged.
    """
    m = matches.sort_values(["match_date", "match_id"]).copy()
    elo: dict[int, float] = {}
    pre_home, pre_away = [], []

    # Innings totals/ wickets for margin calculation.
    first = totals[totals["innings"] == 1].set_index("match_id")
    second = totals[totals["innings"] == 2].set_index("match_id")

    for row in m.itertuples(index=False):
        h, a = int(row.home_team_id), int(row.away_team_id)
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        pre_home.append(eh)
        pre_away.append(ea)

        winner = row.win_team_id
        if pd.isna(winner) or int(winner) == 0 or int(winner) not in (h, a):
            continue  # no result / abandoned

        # Expected score for the home side.
        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        score_h = 1.0 if int(winner) == h else 0.0

        # Margin multiplier.
        if int(winner) == h:
            bat_first, bat_second = h, a
        else:
            bat_first, bat_second = a, h
        margin = 1.0
        if row.match_id in first.index and row.match_id in second.index:
            f, s = first.loc[row.match_id], second.loc[row.match_id]
            if int(f["batting_team_id"]) == bat_first:
                margin = max(1.0, f["runs"] - s["runs"])
            else:
                margin = max(1.0, 10.0 - s["wickets"])
        mult = 1.0 + 0.15 * np.log1p(float(margin))

        k = 32.0
        delta = k * mult * (score_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    m["elo_home"] = pre_home
    m["elo_away"] = pre_away
    return m.drop(columns=[c for c in ("index",) if c in m.columns])


def _venue_par(matches: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Expanding, prior-shrunk mean first-innings score per venue."""
    m = matches.sort_values(["match_date", "match_id"]).copy()
    first = totals[totals["innings"] == 1][["match_id", "runs"]]
    m = m.merge(first, on="match_id", how="left")

    global_prior = 160.0
    par_values, running_global = [], []
    g_sum, g_n = 0.0, 0
    for row in m.itertuples(index=False):
        prior = global_prior if g_n == 0 else g_sum / g_n
        running_global.append(prior)
        g_sum += 0.0 if pd.isna(row.runs) else float(row.runs)
        g_n += 0 if pd.isna(row.runs) else 1

    m["_global_par"] = running_global
    m["_venue_key"] = m["venue_id"].astype(str) + "|" + m["format"].astype(str)

    # Expanding mean per venue, shifted so the current match is excluded.
    m = m.sort_values(["match_date", "match_id"])
    grp = m.groupby("_venue_key", sort=False)["runs"]
    m["_venue_prior"] = grp.transform(lambda s: s.shift(1).expanding().mean())
    m["_venue_n"] = grp.transform(lambda s: s.shift(1).expanding().count())
    # Shrink venue mean toward the global prior with a 15-match prior.
    m["venue_par"] = np.where(
        m["_venue_prior"].notna(),
        (m["_venue_prior"] * m["_venue_n"] + m["_global_par"] * 15.0) / (m["_venue_n"] + 15.0),
        m["_global_par"],
    )
    return m


def build_match_info(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per match with format, stakes, Elo, venue par and totals."""
    m = raw["matches"].copy()
    comps = raw["competitions"][["comp_id", "comp_name", "match_format"]]
    m = m.merge(comps, on="comp_id", how="left")
    m["format"] = [
        _format_from_comp(cn, mf) for cn, mf in zip(m["comp_name"], m["match_format"])
    ]
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
    m["is_ipl"] = (m["format"] == "IPL").astype(int)

    # ---- Innings totals from deliveries (authoritative, includes extras) ----
    b = raw["ball_by_ball"].copy()
    b["ball_runs"] = (
        b["runs"].fillna(0) + b["extras1_runs"].fillna(0) + b["extras_2_runs"].fillna(0)
    )
    totals = (
        b.groupby(["match_id", "innings"])
        .agg(
            runs=("ball_runs", "sum"),
            wickets=("is_wicket", "sum"),
            batting_team_id=("batting_team_id", "first"),
        )
        .reset_index()
    )

    m = _process_elo(m, totals)
    m = _venue_par(m, totals)

    # ---- Winner / result flags ----
    m["no_result"] = (
        m["win_team_id"].isna()
        | (m["win_team_id"].fillna(0).astype(int) == 0)
        | m["result"].astype(str).str.contains("No Result|Abandoned|abandoned", regex=True, na=False)
    )
    m["stakes"] = m["stage"].map(STAGE_STAKES).fillna(1.0)
    m.loc[(m["knock_out"] == 1) & (m["stakes"] < 1.10), "stakes"] = 1.15

    # ---- Internal consistency of the reconstructed chase ----
    first = totals[totals["innings"] == 1].set_index("match_id")
    second = totals[totals["innings"] == 2].set_index("match_id")
    m = m.set_index("match_id")
    target = np.where(
        m["revised_target"].fillna(0) > 0,
        m["revised_target"],
        first["runs"].reindex(m.index).to_numpy() + 1,
    )
    chase_score = second["runs"].reindex(m.index).to_numpy()
    chase_winner = second["batting_team_id"].reindex(m.index).to_numpy()
    chasing_won = m["win_team_id"].to_numpy() == chase_winner
    consistent = np.where(chasing_won, chase_score >= target, chase_score < target)
    m["state_consistent"] = consistent | m["no_result"].to_numpy()
    m = m.reset_index()

    m["ump_overs_1"] = m["first_inn_revised_overs"].replace(0, np.nan).fillna(20).clip(1, 50)
    m["ump_overs_2"] = m["second_inn_revised_overs"].replace(0, np.nan).fillna(20).clip(1, 50)

    # Match-level helper only; the ball table recomputes the label per innings
    # (the batting side changes between innings).
    m["home_team_won"] = np.where(
        m["no_result"], np.nan, (m["win_team_id"] == m["home_team_id"]).astype(float)
    )
    return m


# ---------------------------------------------------------------------------
# Ball-level state
# ---------------------------------------------------------------------------

def build_ball_state(raw: dict[str, pd.DataFrame], match_info: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-delivery state to every ball plus post-delivery state columns."""
    b = raw["ball_by_ball"].copy()
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    b["is_legal"] = ~b["extras_type"].isin(ILLEGAL_DELIVERIES)
    b["ball_runs"] = b["runs"].fillna(0) + b["extras1_runs"].fillna(0) + b["extras_2_runs"].fillna(0)
    b["bat_runs"] = b["runs"].fillna(0)
    b["is_wicket"] = b["is_wicket"].astype(bool)

    g = b.groupby(["match_id", "innings"], sort=False)

    # ---- Pre-delivery state ----
    b["runs_before"] = g["ball_runs"].cumsum() - b["ball_runs"]
    b["wkts_before"] = g["is_wicket"].cumsum() - b["is_wicket"].astype(int)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["ball_in_innings"] = b["legal_before"] + 1

    # ---- Batter state before the ball ----
    sb = b.groupby(["match_id", "innings", "striker_id"], sort=False)
    b["striker_runs_before"] = sb["bat_runs"].cumsum() - b["bat_runs"]
    b["striker_balls_before"] = sb["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["striker_dismissed_before"] = sb["is_wicket"].cumsum() - b["is_wicket"].astype(int)

    # ---- Post-delivery state (for delta win probability) ----
    b["runs_after"] = b["runs_before"] + b["ball_runs"]
    b["wkts_after"] = b["wkts_before"] + b["is_wicket"].astype(int)
    b["legal_after"] = b["legal_before"] + b["is_legal"].astype(int)

    # ---- Innings context ----
    b["phase"] = pd.cut(
        b["overid"], bins=[0, 6, 15, 50], labels=["powerplay", "middle", "death"]
    ).astype(str)

    # ---- Join match context ----
    ctx = match_info[
        [
            "match_id", "format", "is_ipl", "stakes", "knock_out", "stage",
            "venue_id", "venue_country", "venue_par", "home_team_id", "away_team_id",
            "win_team_id", "no_result", "elo_home", "elo_away", "ump_overs_1", "ump_overs_2",
            "match_date", "state_consistent",
        ]
    ]
    b = b.merge(ctx, on="match_id", how="left", suffixes=("", "_match"))

    # Elo of batting / bowling side.
    b["bat_elo"] = np.where(
        b["batting_team_id"] == b["home_team_id"], b["elo_home"], b["elo_away"]
    )
    b["bowl_elo"] = np.where(
        b["bowling_team_id"] == b["home_team_id"], b["elo_home"], b["elo_away"]
    )

    # ---- Target / overs context per innings ----
    inn_tot = (
        b.groupby(["match_id", "innings"])
        .agg(innings_runs=("ball_runs", "sum"), innings_wkts=("is_wicket", "sum"))
        .reset_index()
    )
    inn_tot["next_target"] = inn_tot["innings_runs"] + 1
    tgt = inn_tot[["match_id", "innings", "next_target"]].rename(
        columns={"innings": "prev_innings"}
    )
    tgt["innings"] = tgt["prev_innings"] + 1

    b = b.merge(tgt[["match_id", "innings", "next_target"]], on=["match_id", "innings"], how="left")

    revised = match_info.set_index("match_id")["revised_target"]
    b["revised_target"] = b["match_id"].map(revised)
    b["target"] = np.where(
        b["innings"] == 2,
        np.where(b["revised_target"].fillna(0) > 0, b["revised_target"], b["next_target"]),
        np.where(b["innings"] == 4, b["next_target"], np.nan),
    )
    b["innings_overs"] = np.where(
        b["innings"].isin([1, 3]), b["ump_overs_1"], b["ump_overs_2"]
    )
    b["is_super_over"] = b["innings"].isin([3, 4])

    # ---- Derived chase / scoring state ----
    balls_total = b["innings_overs"] * 6
    b["balls_remaining"] = np.maximum(0.0, balls_total - b["legal_before"])
    b["crr"] = np.where(b["legal_before"] > 0, b["runs_before"] * 6.0 / b["legal_before"], 0.0)
    b["rrr"] = np.where(
        (b["innings"].isin([2, 4])) & (b["balls_remaining"] > 0),
        (b["target"] - b["runs_before"]) * 6.0 / b["balls_remaining"],
        np.nan,
    )
    b["runs_needed"] = np.where(b["innings"].isin([2, 4]), b["target"] - b["runs_before"], np.nan)
    # Per-innings label: did the side batting now win the match?
    b["batting_team_won"] = np.where(
        b["no_result"].astype(bool), np.nan,
        (b["win_team_id"].astype(float) == b["batting_team_id"].astype(float)).astype(float),
    )
    b["projected_score"] = np.where(
        b["legal_before"] > 0, b["runs_before"] * balls_total / b["legal_before"], 0.0
    )

    # ---- Personal pressure ingredients ----
    b["batter_nervous_nineties"] = b["striker_runs_before"].between(90, 99)
    b["batter_new"] = b["striker_balls_before"] < 6
    b["batter_set"] = b["striker_runs_before"] >= 50
    b["bowler_final_over"] = (
        (b["overid"] >= b["innings_overs"].astype(int))
        & b["innings"].isin([2, 4])
        & (b["balls_remaining"] <= 6)
    )
    # Hat-trick: the two previous deliveries by this bowler in this innings were wickets.
    b["_prev_wkt1"] = b.groupby(["match_id", "innings"])["is_wicket"].shift(1).fillna(False)
    b["_prev_wkt2"] = b.groupby(["match_id", "innings"])["is_wicket"].shift(2).fillna(False)
    b["_prev_bowler1"] = b.groupby(["match_id", "innings"])["bowler_id"].shift(1)
    b["_prev_bowler2"] = b.groupby(["match_id", "innings"])["bowler_id"].shift(2)
    b["bowler_hat_trick"] = (
        b["_prev_wkt1"] & b["_prev_wkt2"]
        & (b["_prev_bowler1"] == b["bowler_id"])
        & (b["_prev_bowler2"] == b["bowler_id"])
    )

    # ---- Dismissal attribution ----
    b["striker_out"] = b["is_wicket"] & (
        b["dismissed_player_id"].fillna(-1).astype(float) == b["striker_id"].astype(float)
    )
    b["nonstriker_runout"] = (
        b["is_wicket"]
        & (b["dismissal_type"] == "RunOut")
        & (b["dismissed_player_id"].fillna(-1).astype(float) == b["non_striker_id"].astype(float))
    )
    b["bowler_credited_wicket"] = b["is_wicket"] & b["dismissal_type"].isin(
        ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket", "Obstracting the field", "Obstructing the field"]
    )
    b["non_bowler_wicket"] = b["is_wicket"] & b["dismissal_type"].isin(["RunOut", "Retire Out"])

    # Convenience display fields.
    b["is_four"] = b["is_four"].astype(bool)
    b["is_six"] = b["is_six"].astype(bool)
    return b


def build_state() -> pd.DataFrame:
    raw = load_raw()
    match_info = build_match_info(raw)
    balls = build_ball_state(raw, match_info)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    balls.to_parquet(OUTPUT_DIR / "pressure_balls_state.parquet", index=False)
    match_info.to_parquet(OUTPUT_DIR / "match_info.parquet", index=False)
    return balls


if __name__ == "__main__":
    df = build_state()
    print(f"ball state rows: {len(df):,}")
    print(df[["runs_before", "wkts_before", "legal_before", "target", "rrr"]].describe().to_string())
