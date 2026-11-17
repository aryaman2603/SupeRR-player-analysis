"""Central configuration for the pressure-performance pipeline.

Every tunable lives here so the ranking can be reproduced and stress-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "report"

# Deliveries that do not count toward the six legal balls of an over.
# `other` is the dataset's no-ball variant (extras1_runs == 1 and the over extends).
ILLEGAL_DELIVERIES = ("wide", "Overstep", "other", "Penalty")

# Dismissal types credited to the bowler.
BOWLER_WICKETS = (
    "Caught", "Bowled", "LBW", "Stump", "Hit Wicket",
    "Obstracting the field", "Obstructing the field",
)
# Dismissals that should not be charged to the bowler's pressure ledger.
NON_BOWLER_WICKETS = ("RunOut", "Retire Out")

STAGE_STAKES = {
    "League": 1.00,
    "Others": 1.00,
    "Super 4": 1.05,
    "Qualifier": 1.15,
    "Eliminator": 1.15,
    "Semi Final": 1.20,
    "Final": 1.30,
}


@dataclass(frozen=True)
class Params:
    """All modelling and ranking knobs."""

    seed: int = 42

    # ---- Pressure Index component weights (sum to 1.00) ----
    w_leverage: float = 0.35
    w_chase: float = 0.20
    w_wickets: float = 0.15
    w_phase: float = 0.10
    w_personal: float = 0.10

    # ---- Personal-pressure increments (0-1 scale) ----
    p_nervous_nineties: float = 0.50
    p_new_batter: float = 0.30
    p_set_batter: float = 0.20
    p_final_over: float = 0.50
    p_hat_trick: float = 0.70

    # ---- Scoring ----
    runs_weight: float = 0.80               # composite weight on runs above expectation
    wickets_weight: float = 0.20            # composite weight on wickets above expectation
    gamma: float = 1.0                      # pressure emphasis inside the pressure sample
    format_weights: dict = field(default_factory=lambda: {"IPL": 1.0, "T20I": 0.6})
    shrinkage_k: float = 150.0              # pseudo-balls pull toward own overall level

    # Pressure sample: the top ~20% of deliveries by Pressure Index.
    pressure_sample_cut: float = 40.0
    min_pressure_balls_bat: int = 150        # raw pressure deliveries faced
    min_pressure_balls_bowl: int = 150       # raw pressure deliveries bowled
    min_innings: int = 18

    # Display bands.
    pressure_cut: float = 48.0               # top decile, for "high pressure" stats
    calm_cut: float = 25.0                   # bottom third

    # ---- Bootstrap ----
    bootstrap_reps: int = 500
    bootstrap_ci: float = 0.90

    # ---- Index display ----
    index_center: float = 50.0
    index_scale: float = 15.0

    # ---- Win-probability model ----
    wp_max_iter: int = 350
    wp_learning_rate: float = 0.05
    wp_max_leaf_nodes: int = 31
    wp_min_samples_leaf: int = 80
    wp_l2: float = 1.0
    n_folds: int = 5


DEFAULT_PARAMS = Params()

# Sensitivity grid used in the reliability analysis.
SENSITIVITY = {
    "gamma": [0.5, 1.0, 2.0],
    "shrinkage_k": [100.0, 150.0, 200.0],
    "t20i_weight": [0.5, 0.6, 0.75],
    "wicket_weight": [0.0, 0.2, 0.4],
    "pressure_cut": [35.0, 40.0, 45.0],
}
