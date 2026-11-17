"""Generate every chart used in the competition deck."""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from .config import DEFAULT_PARAMS, FIGURE_DIR, OUTPUT_DIR
from .validation import _aggregate, _prepare

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#B8C4CC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

INK = "#12232E"
TEAL = "#0B6E75"
TEAL_LIGHT = "#5FA8A8"
ORANGE = "#D9644A"
GOLD = "#E9B44C"
GRAY = "#8D99AE"
LIGHT = "#EDF2F4"

PARAMS = DEFAULT_PARAMS


def _save(fig, name: str, dpi: int = 220) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Win-probability calibration
# ---------------------------------------------------------------------------

def fig_wp_calibration() -> None:
    cal = pd.read_csv(OUTPUT_DIR / "wp_calibration.csv")
    metrics = json.loads((OUTPUT_DIR / "wp_metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.4, ls="--", label="Perfect calibration")
    sizes = 18 + 260 * (cal["n"] / cal["n"].max())
    ax.scatter(cal["predicted"], cal["observed"], s=sizes, color=TEAL, alpha=0.85,
               edgecolor="white", linewidth=1.2, zorder=3, label="Out-of-fold deciles")
    for _, r in cal.iterrows():
        ax.plot([r["predicted"], r["predicted"]], [r["predicted"], r["observed"]],
                color=TEAL, alpha=0.25, lw=1)

    m1, m2 = metrics["innings_1"], metrics["innings_2"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("The win-probability model is calibrated out-of-fold")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.98, 0.04,
        f"Innings 1 Brier {m1['brier']:.3f} (baseline {m1['baseline_brier']:.3f})\n"
        f"Chases Brier {m2['brier']:.3f} (baseline {m2['baseline_brier']:.3f})",
        ha="right", va="bottom", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="none"),
    )
    _save(fig, "wp_calibration.png")


# ---------------------------------------------------------------------------
# 2. Pressure Index distribution
# ---------------------------------------------------------------------------

def fig_pressure_distribution(balls: pd.DataFrame) -> None:
    d = balls.loc[balls["wp_usable"], "pressure_index"].dropna()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bins = np.linspace(0, 100, 51)
    ax.hist(d[d < 40], bins=bins, color=GRAY, alpha=0.55, label="Routine (<40)")
    ax.hist(d[(d >= 40) & (d < 48)], bins=bins, color=TEAL_LIGHT, alpha=0.9, label="Elevated (40-48)")
    ax.hist(d[d >= 48], bins=bins, color=ORANGE, alpha=0.95, label="High pressure (>=48)")

    ax.axvline(40, color=INK, lw=1.2, ls="--")
    ax.axvline(48, color=INK, lw=1.2, ls=":")
    ax.text(40.8, ax.get_ylim()[1] * 0.92, "pressure sample\n(top 20%)", fontsize=9, color=INK)
    ax.text(48.8, ax.get_ylim()[1] * 0.70, "top decile", fontsize=9, color=INK)
    ax.set_xlabel("Pressure Index (0-100)")
    ax.set_ylabel("Deliveries")
    ax.set_title("Most deliveries are routine; pressure is concentrated")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.99, 0.62,
        f"{len(d):,} deliveries\nmedian {d.median():.0f}  |  90th pct {d.quantile(.9):.0f}  |  99th {d.quantile(.99):.0f}",
        transform=ax.transAxes, ha="right", fontsize=9, color=GRAY,
    )
    _save(fig, "pressure_distribution.png")


# ---------------------------------------------------------------------------
# 3. Pressure by phase and innings type
# ---------------------------------------------------------------------------

def fig_pressure_by_situation(balls: pd.DataFrame) -> None:
    d = balls[balls["wp_usable"]].copy()
    d["innings_type"] = np.where(d["innings"].isin([2, 4]), "Chasing", "Setting")
    order = ["powerplay", "middle", "death"]
    labels = ["Powerplay\n(1-6)", "Middle\n(7-15)", "Death\n(16-20)"]
    pivot = (
        d.pivot_table(index="phase", columns="innings_type", values="pressure_index", aggfunc="mean")
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2, pivot["Setting"], w, color=TEAL, label="Setting (innings 1)")
    b2 = ax.bar(x + w / 2, pivot["Chasing"], w, color=ORANGE, label="Chasing (innings 2)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Pressure Index")
    ax.set_ylim(0, max(pivot.max()) * 1.28)
    ax.set_title("Pressure peaks in death-over chases")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "pressure_by_situation.png")


# ---------------------------------------------------------------------------
# 4. Top-10 index charts
# ---------------------------------------------------------------------------

def _top_chart(df: pd.DataFrame, title: str, name: str) -> None:
    d = df.head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = [ORANGE if i >= 75 else TEAL if i >= 60 else TEAL_LIGHT for i in d["score_index"]]
    bars = ax.barh(d["player_name"], d["score_index"], color=colors, height=0.68)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2,
                f"{r['score_index']:.0f}", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(2.0, b.get_y() + b.get_height() / 2,
                f"90% rank {int(r['rank_lo'])}-{int(r['rank_hi'])}",
                va="center", fontsize=8.2, color="white" if b.get_width() > 22 else GRAY)
    ax.axvline(50, color=GRAY, lw=1.3, ls="--")
    ax.text(50.8, -0.8, "average = 50", fontsize=8.5, color=GRAY)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pressure Performance Index (0-100)")
    ax.set_title(title, pad=12)
    ax.grid(axis="y", visible=False)
    _save(fig, name)


def fig_top_batters() -> None:
    _top_chart(pd.read_csv(OUTPUT_DIR / "top10_batters.csv"),
               "Top 10 batters under pressure", "top_batters.png")


def fig_top_bowlers() -> None:
    _top_chart(pd.read_csv(OUTPUT_DIR / "top10_bowlers.csv"),
               "Top 10 bowlers under pressure", "top_bowlers.png")


# ---------------------------------------------------------------------------
# 5. Pressure vs overall
# ---------------------------------------------------------------------------

def fig_pressure_vs_overall(bat: pd.DataFrame, bowl: pd.DataFrame) -> None:
    """Dumbbell chart: overall level vs pressure level for the top 10."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    for ax, df, title in [
        (axes[0], bat, "Batters"),
        (axes[1], bowl, "Bowlers"),
    ]:
        d = df.sort_values("score_index", ascending=False).head(10).copy()
        d = d.sort_values("prae", ascending=True)
        y = np.arange(len(d))
        overall = d["own_runs"] * 100
        pressure = d["prae"] * 100
        ax.hlines(y, overall, pressure, color="#CBD5DB", lw=3.2, zorder=1)
        ax.scatter(overall, y, s=70, color=GRAY, edgecolor="white", linewidth=1.0,
                   zorder=3, label="Overall level")
        ax.scatter(pressure, y, s=88, color=ORANGE, edgecolor="white", linewidth=1.0,
                   zorder=4, label="Pressure level")
        for yi, (o, p_) in enumerate(zip(overall, pressure)):
            gap = p_ - o
            ax.text(max(o, p_) + 1.6, yi, f"{gap:+.0f}", va="center", fontsize=8.4,
                    color=ORANGE if gap > 0 else GRAY, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(d["player_name"], fontsize=9.5)
        ax.axvline(0, color=INK, lw=0.9, alpha=0.35)
        ax.set_xlabel("Runs above expectation / 100 balls")
        ax.set_title(title, pad=10)
        ax.grid(axis="y", visible=False)
        lo = min(overall.min(), pressure.min()) - 6
        hi = max(overall.max(), pressure.max()) + 10
        ax.set_xlim(lo, hi)

    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Pressure level vs overall level: who lifts, who holds",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "pressure_vs_overall.png")


# ---------------------------------------------------------------------------
# 6. Phase heatmaps
# ---------------------------------------------------------------------------

def fig_phase_heatmap(bat: pd.DataFrame, bowl: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    cols = ["phase_powerplay", "phase_middle", "phase_death"]
    labels = ["Powerplay", "Middle", "Death"]

    for ax, df, title in [(axes[0], bat, "Batters"), (axes[1], bowl, "Bowlers")]:
        d = df.sort_values("score_index", ascending=False).head(10).set_index("player_name")[cols]
        d.columns = labels
        vmax = np.nanmax(np.abs(d.values))
        im = ax.imshow(d.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.index, fontsize=9)
        ax.set_title(title)
        ax.grid(False)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8.4,
                        color="white" if abs(v) > vmax * 0.55 else INK)

    fig.subplots_adjust(left=0.12, right=0.88, wspace=0.55, top=0.84, bottom=0.10)
    cax = fig.add_axes([0.90, 0.16, 0.018, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Runs above expected per 100 pressure balls")
    fig.suptitle("Phase fingerprints: where each player earns the score", fontsize=16, fontweight="bold", y=1.01)
    _save(fig, "phase_heatmap.png")


# ---------------------------------------------------------------------------
# 7. Sensitivity
# ---------------------------------------------------------------------------

def fig_sensitivity() -> None:
    s = pd.read_csv(OUTPUT_DIR / "sensitivity_summary.csv").copy()
    baseline = {"gamma=1.0", "shrinkage_k=150", "t20i_weight=0.6", "wicket_weight=0.2", "pressure_cut=40"}
    s = s[~s["variant"].isin(baseline)].copy()
    s["mean_overlap"] = (s["bat_top10_overlap"] + s["bowl_top10_overlap"]) / 2
    s = s.sort_values("mean_overlap")

    pretty = {
        "gamma=0.5": "Pressure emphasis 0.5",
        "gamma=2.0": "Pressure emphasis 2.0",
        "shrinkage_k=100": "Shrinkage k = 100",
        "shrinkage_k=200": "Shrinkage k = 200",
        "t20i_weight=0.5": "T20I weight 0.5",
        "t20i_weight=0.75": "T20I weight 0.75",
        "wicket_weight=0.0": "Wickets weight 0.0",
        "wicket_weight=0.4": "Wickets weight 0.4",
        "pressure_cut=35": "Pressure cut 35",
        "pressure_cut=45": "Pressure cut 45",
        "pi_leverage_heavy": "PI: leverage-heavy",
        "pi_situation_heavy": "PI: situation-heavy",
    }
    s["label"] = s["variant"].map(pretty).fillna(s["variant"])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = np.arange(len(s))
    ax.hlines(y, s["bat_top10_overlap"], s["bowl_top10_overlap"], color="#C9D3D9", lw=3, zorder=1)
    ax.scatter(s["bat_top10_overlap"], y, color=TEAL, s=58, label="Batters", zorder=3)
    ax.scatter(s["bowl_top10_overlap"], y, color=ORANGE, s=58, label="Bowlers", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(s["label"], fontsize=9)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Top-10 names retained vs baseline (out of 10)")
    ax.set_title("Stable to most choices; membership shifts only at extreme cuts")
    ax.axvline(8, color=GRAY, ls="--", lw=1.1)
    ax.text(8.1, 0.2, "8/10", fontsize=8.5, color=GRAY)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "sensitivity.png")


# ---------------------------------------------------------------------------
# 8. Case study: Rinku's last-over chase
# ---------------------------------------------------------------------------

def fig_case_study(balls: pd.DataFrame, match_info: pd.DataFrame) -> None:
    mid, innings = 20193, 2
    d = balls[(balls["match_id"] == mid) & (balls["innings"] == innings)].copy()
    d = d.sort_values("innings_ball_num").reset_index(drop=True)
    d["ball_no"] = np.arange(1, len(d) + 1)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    ax.plot(d["ball_no"], d["wp_before"], color=TEAL, lw=2.0, zorder=3)
    ax.fill_between(d["ball_no"], 0, d["wp_before"], color=TEAL, alpha=0.08)
    ax.axhline(0.5, color=GRAY, ls="--", lw=1)
    wk = d[d["is_wicket"]]
    ax.scatter(wk["ball_no"], wk["wp_before"], marker="x", s=44, color=GRAY, zorder=4)
    last_over_start = d[d["overid"] == d["overid"].max()]["ball_no"].min()
    ax.axvspan(last_over_start - 0.5, d["ball_no"].max() + 0.5, color=GOLD, alpha=0.15, zorder=0)
    ax.annotate("Final over", (last_over_start + 0.5, 0.55), fontsize=9, color=ORANGE)
    ax.set_xlabel("Delivery in the chase")
    ax.set_ylabel("KKR win probability")
    ax.set_ylim(0, 1)
    ax.set_title("The whole chase: one team was nearly out")

    # Last over: six balls, 29 needed.
    lo = d[d["overid"] == d["overid"].max()].copy()
    lo["ball"] = np.arange(1, len(lo) + 1)
    ax2.plot(lo["ball"], lo["wp_before"], color=TEAL, lw=2.2, marker="o", ms=7, zorder=3)
    ax2.axhline(0.5, color=GRAY, ls="--", lw=1)
    outcomes = []
    for _, r in lo.iterrows():
        outcomes.append("6" if r["is_six"] else str(int(r["bat_runs"])))
    for i, (_, r) in enumerate(lo.iterrows()):
        ax2.annotate(outcomes[i], (r["ball"], r["wp_before"]), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=ORANGE if outcomes[i] == "6" else INK, fontweight="bold")
    # Winning moment.
    ax2.scatter([len(lo) + 1], [1.0], marker="*", s=220, color=GOLD,
                edgecolor=ORANGE, linewidth=0.8, zorder=5)
    ax2.plot([len(lo), len(lo) + 1], [lo["wp_before"].iloc[-1], 1.0],
             color=ORANGE, lw=2, ls="--", zorder=4)
    ax2.annotate("Won", (len(lo) + 1, 1.0), xytext=(0, -16), textcoords="offset points",
                 ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    ax2.set_xticks(list(lo["ball"]) + [len(lo) + 1])
    ax2.set_xticklabels([str(b) for b in lo["ball"]] + ["win"])
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("Ball of the final over")
    ax2.set_ylabel("KKR win probability before ball")
    ax2.set_title("29 needed off 6: the last over")

    fig.suptitle("Case study: Rinku Singh, 48* off 21 - KKR vs Gujarat Titans, 9 Apr 2023",
                 fontsize=15.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "case_study.png")


# ---------------------------------------------------------------------------
# 9. Split-half reliability
# ---------------------------------------------------------------------------

def fig_reliability(balls: pd.DataFrame) -> None:
    rng = np.random.default_rng(PARAMS.seed)
    matches = balls["match_id"].unique()
    sel = set(rng.choice(matches, len(matches) // 2, replace=False))
    means = json.loads((OUTPUT_DIR / "validation.json").read_text())["split_half"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    for ax, role, title, color in [
        (axes[0], "batter", "Batters", TEAL),
        (axes[1], "bowler", "Bowlers", ORANGE),
    ]:
        d = _prepare(balls, role)
        a = _aggregate(d[d["match_id"].isin(sel)])
        c = _aggregate(d[~d["match_id"].isin(sel)])
        common = a.index.intersection(c.index)
        mean_rho = means[role]["spearman"]
        ax.scatter(a.loc[common, "composite"], c.loc[common, "composite"], s=46,
                   color=color, alpha=0.7, edgecolor="white", linewidth=0.7)
        ax.set_xlabel("Score, random half A")
        ax.set_ylabel("Score, random half B")
        ax.set_title(f"{title}: mean split-half rho = {mean_rho:.2f}")
    fig.suptitle("Repeatability check: same players, different matches",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "reliability.png")


def build_all() -> None:
    balls = pd.read_parquet(OUTPUT_DIR / "pressure_balls.parquet")
    bat = pd.read_csv(OUTPUT_DIR / "batter_scores.csv")
    bowl = pd.read_csv(OUTPUT_DIR / "bowler_scores.csv")
    match_info = pd.read_parquet(OUTPUT_DIR / "match_info.parquet")

    fig_wp_calibration()
    fig_pressure_distribution(balls)
    fig_pressure_by_situation(balls)
    fig_top_batters()
    fig_top_bowlers()
    fig_pressure_vs_overall(bat, bowl)
    fig_phase_heatmap(bat, bowl)
    fig_sensitivity()
    fig_case_study(balls, match_info)
    fig_reliability(balls)
    print(f"figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    build_all()
