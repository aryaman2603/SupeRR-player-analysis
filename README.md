# Who thrives when it matters?

A situation-adjusted ranking of the **top 10 batters and top 10 bowlers under pressure**
in the IPL and T20I from 2022 to 2026, built from the six supplied CSVs.

The core idea is simple: **score the situation first, then measure what the player did
relative to an average player in that same situation.** Reputation and raw totals never
enter the ranking.

## Deliverables

| Item | Location |
|---|---|
| Competition deck (13 slides) | [`report/Pressure_Performance_Deck.pptx`](report/Pressure_Performance_Deck.pptx) |
| Deck builder | [`report/build_deck.py`](report/build_deck.py) |
| Full player tables | `outputs/batter_scores.csv`, `outputs/bowler_scores.csv` |
| Top-10 tables | `outputs/top10_batters.csv`, `outputs/top10_bowlers.csv` |
| Charts | `outputs/figures/` |
| Validation report | `outputs/validation.json`, `outputs/wp_metrics.json` |
| Reproducible pipeline | `src/pressure/pipeline.py` |

## Headline results

Index is 0-100 with **50 = average eligible player**. Brackets are 90% match-level
bootstrap rank intervals.

### Top 10 batters

| # | Player | Index | 90% rank | Inns | Pressure balls | Pressure SR | RAE / 100 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Rinku Singh | 81.2 | 1-17 | 83 | 262 | 212.5 | +41 |
| 2 | Tim David | 76.2 | 1-25 | 98 | 347 | 188.3 | +29 |
| 3 | Ashutosh Sharma | 72.2 | 1-33 | 25 | 172 | 204.1 | +35 |
| 4 | Yashasvi Jaiswal | 71.3 | 1-23 | 88 | 158 | 159.6 | +41 |
| 5 | Dinesh Karthik | 68.7 | 1-36 | 61 | 152 | 190.7 | +33 |
| 6 | Rovman Powell | 68.1 | 2-27 | 100 | 388 | 207.4 | +30 |
| 7 | David Miller | 67.1 | 2-33 | 81 | 367 | 200.0 | +28 |
| 8 | Rashid Khan | 65.6 | 2-48 | 71 | 310 | 160.3 | +23 |
| 9 | Heinrich Klaasen | 64.1 | 4-37 | 72 | 208 | 181.9 | +25 |
| 10 | Suryakumar Yadav | 64.0 | 4-34 | 137 | 288 | 158.5 | +21 |

### Top 10 bowlers

| # | Player | Index | 90% rank | Inns | Pressure balls | Pressure econ | Wkts | RAE / 100 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Sunil Narine | 84.4 | 1-13 | 66 | 205 | 5.81 | 8 | +39 |
| 2 | Mohsin Khan | 82.5 | 1-27 | 30 | 165 | 5.52 | 11 | +41 |
| 3 | Rashid Khan | 80.6 | 1-24 | 122 | 490 | 7.65 | 27 | +27 |
| 4 | Kuldeep Yadav | 79.7 | 1-17 | 88 | 329 | 8.33 | 12 | +33 |
| 5 | Jasprit Bumrah | 76.0 | 2-38 | 74 | 355 | 7.23 | 13 | +30 |
| 6 | Mohammad Nawaz | 72.6 | 2-30 | 52 | 238 | 6.00 | 16 | +23 |
| 7 | Krunal Pandya | 72.2 | 1-43 | 66 | 243 | 6.69 | 11 | +26 |
| 8 | Abrar Ahmed | 70.5 | 2-36 | 35 | 206 | 4.88 | 8 | +29 |
| 9 | Maheesh Theekshana | 69.0 | 5-35 | 98 | 407 | 6.83 | 14 | +29 |
| 10 | Yuzvendra Chahal | 68.5 | 3-39 | 97 | 467 | 8.39 | 24 | +18 |

For bowlers, **RAE / 100 = runs saved above expectation per 100 pressure balls**
(positive is good).

## Data scope

- IPL + T20I only, 2022-2026. The source files also contain 30 other T20 matches
  (Emerging Teams Asia Cup / Rising Stars); these are filtered out in `load_raw`.
- **837 matches**, **191,668** usable legal deliveries, **39,144** pressure deliveries
  (PI >= 40), **19,342** high-pressure deliveries (PI >= 48).
- Excluded: no-result matches, 2 DLS-inconsistent matches (IDs 24822 and 28334),
  and super-over innings.
- 63 batters and 86 bowlers pass the minimum-exposure filter.

## Method

### 1. Pressure Index (situation)

Every delivery is scored **before** the outcome:

```
PI = 100 × clip( S × (0.35·L + 0.20·C + 0.15·W + 0.10·P + 0.10·M), 0, 1 )
```

| Component | Weight | Meaning |
|---|---:|---|
| `L` leverage | 0.35 | Data-driven SD of win-probability change in the exact state, shrunk toward the league mean |
| `C` chase deficit | 0.20 | Runs needed vs balls left, normalised to par; projected-score gap when setting |
| `W` wickets | 0.15 | Wickets lost while setting, wickets in hand while chasing |
| `P` phase | 0.10 | Powerplay 0.3, middle 0.2, death 1.0 |
| `M` personal | 0.10 | Nervous nineties, new batter, set batter, final over, hat-trick ball |
| `S` stakes | multiplier | League 1.0 → final 1.3; knockout floor 1.15 |

The index is left-skewed: median 29, 90th percentile 48. The **pressure sample** is
PI >= 40 (top ~20%); the **high-pressure** display band is PI >= 48 (top decile).

### 2. Win-probability model

Separate logistic regressions for innings 1 and the chase, with features for score,
wickets, balls left, target, required rate, Elo, venue par and format. Predictions are
**out-of-fold**, with `GroupKFold` by match so a match is never in both train and test.

| Model | Brier | Baseline |
|---|---:|---:|
| Innings 1 | 0.218 | 0.250 |
| Chase | 0.130 | 0.249 |

Mean out-of-fold ΔWP is approximately 0. Calibration is shown in
`outputs/figures/wp_calibration.png`.

### 3. Player currency: runs and wickets above expectation

For every pressure delivery, bucket means (phase × wickets × overs × required rate ×
format, falling back to coarser buckets) estimate what an average player would have
produced in that state.

- **Batter:** `runs_above = runs_off_bat − expected_runs`;
  `wkts_above = expected_dismissal_prob − dismissed`.
- **Bowler:** `runs_above = expected_runs_conceded − runs_conceded`;
  `wkts_above = wicket_taken − expected_wicket_prob`.

Runs above expectation are roughly twice as repeatable across random match splits as
raw ΔWP, so they carry **80%** of the composite; wickets carry **20%**.

Within the pressure sample, deliveries are weighted by `(PI/50)^γ` (γ = 1) and by
format (IPL 1.0, T20I 0.6).

### 4. Shrinkage and index

Each player's pressure score is shrunk toward his own all-ball level, itself shrunk
toward the league mean (k = 150). This stops a short hot streak from topping the list.
Eligibility: **≥150 pressure balls** and **≥18 innings**.

```
composite = 0.80 · z(PRAE) + 0.20 · z(PWAE)
index     = 50 + 15 · z(composite), clipped to 0-100
```

`z` is computed within the eligible player pool using a robust (median/MAD) scale.

### 5. Fair comparisons

- Format weights applied only at aggregation; expectations are estimated within format.
- Batters and bowlers are scored separately. Run-outs are never credited to the bowler;
  non-striker run-outs are excluded from the striker's ledger.
- Situational expectations absorb phase, wickets, chase deficit and scoring rate;
  the WP model additionally uses Elo and venue par.
- Stakes are explicit rather than implicit: league 1.0, Super 4 1.05,
  qualifier/eliminator 1.15, semi 1.20, final 1.30.
- Bootstrap resamples **matches**, not balls (500 reps, 90% intervals), preserving
  within-match correlation.

## Validation

| Check | Batters | Bowlers |
|---|---:|---:|
| Split-half Spearman (20 random match splits) | 0.28 | 0.39 |
| Mean Player-of-the-Match percentile | 0.81 | 0.73 |
| Spearman vs conventional pressure stat | 0.40 | 0.38 |

Sensitivity (γ, k, T20I weight, wicket weight, PI weights, pressure cut): rank
correlation with the baseline is 0.95-1.00, and top-10 overlap is 8-10 of 10 for most
variants. Only extreme pressure cuts (PI >= 45) materially change membership; that
trade-off is shown in the deck.

**Read the lists as sets, not a strict ladder.** The 90% rank intervals are wide because
pressure samples are small; the deck reports them rather than hiding them.

## Reproduce

```bash
# full rebuild: data -> WP -> PI -> scores -> bootstrap -> validation -> figures
uv run python -m src.pressure.pipeline

# rebuild only the deck from current outputs
uv run python -m report.build_deck
```

The pipeline is deterministic (seed 42). A full run takes about one minute on a laptop;
the bootstrap is the slow step.

Smoke tests for the generated outputs:

```bash
python -m unittest discover -s tests -v
```

Key modules:

```
src/pressure/
  data.py             state reconstruction, Elo, venue par, scope filters
  wp_model.py         out-of-fold win-probability model
  pressure_index.py   leverage + PI components
  scoring.py          pressure-sample RAE/PWAE, shrinkage, index
  ranking.py          bootstrap rank intervals, sensitivity grid, top-10 tables
  validation.py       split-half, Player-of-the-Match, conventional agreement
  figures.py          all deck charts
  pipeline.py         end-to-end orchestrator
```

## Limitations

- Pressure is inferred from match state; nerves, fatigue and intent are unobserved.
- No ball-tracking or fielding data: placement, catch difficulty and field pressure are missing.
- T20I exposure is thinner and more variable than IPL exposure; format weights only partly fix that.
- The exact order is noisy; use the intervals and the sensitivity analysis.
- The clutch premium (high-pressure minus calm) is descriptive and can be volatile for
  low-volume hitters; it is not part of the primary rank.
- Observational, not causal: role and team context shape the situations a player faces.
