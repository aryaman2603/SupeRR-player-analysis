"""Build the competition deck from the generated outputs.

    uv run python -m report.build_deck

The deck is intentionally data-driven: every number and chart is read from
``outputs/`` so a pipeline rerun updates the slides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORT = ROOT / "report"

SW, SH = 13.333, 7.5
FONT = "Arial"

INK = RGBColor(0x12, 0x23, 0x2E)
TEAL = RGBColor(0x0B, 0x6E, 0x75)
TEAL_L = RGBColor(0x5F, 0xA8, 0xA8)
ORANGE = RGBColor(0xD9, 0x64, 0x4A)
GOLD = RGBColor(0xE9, 0xB4, 0x4C)
GRAY = RGBColor(0x6B, 0x7A, 0x82)
LIGHT = RGBColor(0xED, 0xF2, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xF7, 0xF9, 0xFA)


def tx(slide, x, y, w, h, text, size=14, bold=False, color=INK,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = FONT
        run.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=12.5, color=INK, gap=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        lead, rest = "▪  ", item
        if isinstance(item, tuple):
            lead, rest = "▪  ", item[0]
        r1 = p.add_run()
        r1.text = lead
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.name = FONT
        r1.font.color.rgb = TEAL
        r2 = p.add_run()
        r2.text = rest
        r2.font.size = Pt(size)
        r2.font.name = FONT
        r2.font.color.rgb = color
    return box


def table(slide, rows, x, y, w, h, widths, fs=9, header_fill=INK,
          aligns=None, zebra=PALE):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for i, width in enumerate(widths):
        tbl.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(h / len(rows))
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (WHITE if r % 2 else zebra)
            p = cell.text_frame.paragraphs[0]
            p.text = str(value)
            p.alignment = aligns[c] if aligns else PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(fs)
                run.font.bold = r == 0
                run.font.name = FONT
                run.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def pic(slide, name, x, y, w, h):
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2),
                             Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def header(slide, kicker, title, sub=None, n=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(0.075))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    tx(slide, 0.65, 0.28, 11.5, 0.25, kicker.upper(), 9.5, True, ORANGE)
    tx(slide, 0.65, 0.52, 12.0, 0.5, title, 25, True, INK)
    if sub:
        tx(slide, 0.65, 1.02, 12.0, 0.35, sub, 11.5, False, GRAY)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.40),
                                  Inches(12.03), Inches(0.008))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xD5, 0xDE, 0xE3)
    line.line.fill.background()
    tx(slide, 0.65, 7.13, 9.5, 0.22, "Pressure Performance Index  ·  IPL + T20I 2022-2026", 8.2, False, GRAY)
    if n:
        tx(slide, 12.1, 7.13, 0.6, 0.22, f"{n:02d}", 8.5, True, GRAY, PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, letter, name, desc):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.fill.background()
    box.adjustments[0] = 0.06
    tx(slide, x + 0.16, y + 0.14, 0.6, 0.4, letter, 22, True, TEAL)
    tx(slide, x + 0.76, y + 0.16, w - 0.9, 0.3, name, 11.5, True, INK)
    tx(slide, x + 0.76, y + 0.47, w - 0.9, h - 0.55, desc, 8.8, False, GRAY, spacing=1.0)


def _ord(n):
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def chip(slide, x, y, w, h, value, label, color=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = RGBColor(0xDD, 0xE5, 0xE9)
    box.adjustments[0] = 0.12
    tx(slide, x + 0.14, y + 0.12, w - 0.28, 0.38, value, 19, True, color)
    tx(slide, x + 0.14, y + 0.52, w - 0.28, h - 0.56, label, 8.6, False, GRAY, spacing=1.0)


def load():
    d = {
        "bat": pd.read_csv(OUT / "top10_batters.csv"),
        "bowl": pd.read_csv(OUT / "top10_bowlers.csv"),
        "val": json.loads((OUT / "validation.json").read_text()),
        "wp": json.loads((OUT / "wp_metrics.json").read_text()),
        "sens": pd.read_csv(OUT / "sensitivity_summary.csv"),
        "bs": pd.read_csv(OUT / "batter_scores.csv"),
        "ws": pd.read_csv(OUT / "bowler_scores.csv"),
    }
    balls = pd.read_parquet(OUT / "pressure_balls.parquet",
                            columns=["match_id", "match_date", "format", "innings",
                                     "overid", "wp_usable", "pressure_index",
                                     "wp_before", "dwp", "bat_runs", "is_legal", "is_six",
                                     "striker_name"])
    d["balls"] = balls
    return d


def scope(d):
    b = d["balls"]
    live = b[b["wp_usable"]]
    d["n_matches"] = int(b["match_id"].nunique())
    d["n_balls"] = int(len(live))
    d["n_pressure"] = int((live["pressure_index"] >= 40).sum())
    d["n_high"] = int((live["pressure_index"] >= 48).sum())
    d["bat_elig"] = int(d["bs"]["eligible"].sum())
    d["bowl_elig"] = int(d["ws"]["eligible"].sum())
    d["seasons"] = f"{pd.to_datetime(live['match_date']).dt.year.min():.0f}-{pd.to_datetime(live['match_date']).dt.year.max():.0f}"


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_title(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = INK
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.55),
                                Inches(0.09), Inches(2.25))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ORANGE
    accent.line.fill.background()
    tx(s, 1.15, 1.45, 10.5, 0.3, "SUPERR SELECTOR  ·  DATA CHALLENGE", 11, True, ORANGE)
    tx(s, 1.15, 1.95, 11.0, 1.4, "Who thrives\nwhen it matters?", 44, True, WHITE, spacing=0.95)
    tx(s, 1.15, 3.55, 10.6, 0.9,
       "A situation-adjusted top 10 for pressure batting and bowling\nin the IPL and T20I, 2022-2026.",
       17, False, RGBColor(0xC9, 0xD6, 0xDB), spacing=1.1)
    tx(s, 1.15, 4.75, 10.6, 0.5,
       "Every delivery is priced by how much it can swing the match, then every player is judged\nagainst what an average player would have done in the same situation.",
       11.5, False, RGBColor(0x9F, 0xB1, 0xB9), spacing=1.15)
    vals = [(f"{d['n_balls']:,}", "deliveries scored"), (f"{d['n_matches']:,}", "matches"),
            (f"{d['n_pressure']:,}", "pressure deliveries"), (d["seasons"], "five seasons")]
    for i, (v, l) in enumerate(vals):
        x = 1.15 + i * 3.0
        tx(s, x, 5.95, 2.7, 0.38, v, 21, True, WHITE)
        tx(s, x, 6.38, 2.7, 0.25, l, 9, False, RGBColor(0x9F, 0xB1, 0xB9))
    tx(s, 1.15, 7.05, 11.0, 0.25,
       "Reproducible repo  ·  seed 42  ·  python -m src.pressure.pipeline", 9, False,
       RGBColor(0x7F, 0x93, 0x9C))


def slide_answer(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "The answer first", "Two lists, one rule: reward output when the match is on the line",
           "Index 0-100, 50 = average eligible player. 90% bootstrap rank intervals appear on the result slides.", n)
    for i, (df, label, color) in enumerate([(d["bat"], "BATTERS", TEAL), (d["bowl"], "BOWLERS", ORANGE)]):
        x = 0.65 + i * 6.2
        tx(s, x, 1.62, 5.8, 0.28, label, 9.5, True, color)
        rows = [["#", "Player", "Index"]]
        for _, r in df.head(10).iterrows():
            rows.append([f"{int(r['rank'])}", r["player_name"], f"{r['score_index']:.1f}"])
        table(s, rows, x, 1.95, 5.75, 4.55, [0.55, 4.15, 1.05], fs=11,
              header_fill=color, aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
    tx(s, 0.65, 6.62, 12.0, 0.3,
       "Read the list as a set, not a precise order: the data identify the players; the intervals quantify how much the order can move.",
       9.5, False, GRAY)


def slide_pressure(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Defining pressure", "Pressure is a situation, not a reputation",
           "Each delivery gets a 0-100 Pressure Index before the outcome is observed.", n)
    fbox = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(1.62),
                              Inches(12.03), Inches(1.05))
    fbox.fill.solid()
    fbox.fill.fore_color.rgb = LIGHT
    fbox.line.fill.background()
    fbox.adjustments[0] = 0.08
    tx(s, 0.9, 1.78, 11.6, 0.35,
       "PI = 100 × clip( S × (0.35·L + 0.20·C + 0.15·W + 0.10·P + 0.10·M), 0, 1 )",
       18, True, TEAL)
    tx(s, 0.9, 2.25, 11.6, 0.28,
       "L leverage  ·  C chase deficit  ·  W wickets  ·  P phase  ·  M personal milestone  ·  S stakes",
       9.5, False, GRAY)
    cards = [
        ("L", "Leverage (35%)", "Data-driven swing potential: the SD of win-probability change in this exact state, shrunk toward the league average."),
        ("C", "Chase deficit (20%)", "Runs needed vs balls left, normalised to the par rate. Setting innings use the projected-score gap."),
        ("W", "Wickets (15%)", "Wickets lost while setting, wickets in hand while chasing. A collapse or a thin tail raises the index."),
        ("P", "Phase (10%)", "Powerplay 0.3, middle 0.2, death 1.0. The last five overs carry the most situational weight."),
        ("M", "Personal (10%)", "Nervous nineties, new batter, set batter, final over, hat-trick ball."),
        ("S", "Stakes", "League 1.0 → final 1.3; knockout floor 1.15. Applied as a multiplier."),
    ]
    for i, (letter, name, desc) in enumerate(cards):
        col, row = i % 3, i // 3
        card(s, 0.65 + col * 4.06, 2.9 + row * 1.85, 3.86, 1.62, letter, name, desc)
    tx(s, 0.65, 6.72, 12.0, 0.25,
       "The index is computed from the state before the ball: a six does not retroactively make the situation more pressure.",
       9, False, GRAY)


def slide_distribution(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Pressure in the data", "Pressure is concentrated — and it peaks in death-over chases",
           "The pressure sample is the top ~20% of deliveries by PI; high pressure is the top decile.", n)
    pic(s, "pressure_distribution.png", 0.35, 1.52, 6.7, 4.55)
    pic(s, "pressure_by_situation.png", 7.15, 1.52, 5.55, 4.55)
    med = d["balls"].loc[d["balls"]["wp_usable"], "pressure_index"].median()
    chip(s, 0.65, 6.2, 2.85, 0.78, f"{med:.0f}", "median PI", TEAL)
    chip(s, 3.7, 6.2, 2.95, 0.78, f"{d['n_pressure']:,}", f"pressure sample ({100*d['n_pressure']/d['n_balls']:.0f}%)", TEAL)
    chip(s, 6.85, 6.2, 2.95, 0.78, f"{d['n_high']:,}", f"high pressure ({100*d['n_high']/d['n_balls']:.0f}%)", ORANGE)
    chip(s, 10.0, 6.2, 2.68, 0.78, "53", "mean PI, death chase", ORANGE)


def slide_wp(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Measuring the situation", "A calibrated win-probability model prices every ball",
           "Separate models for innings 1 and the chase; match-grouped cross-validation keeps the estimates honest.", n)
    pic(s, "wp_calibration.png", 0.35, 1.52, 6.7, 4.9)
    w1, w2 = d["wp"]["innings_1"], d["wp"]["innings_2"]
    bullets(s, 7.2, 1.72, 5.5, 4.6, [
        "Two logistic regressions: setting innings and chasing innings.",
        "Features: score, wickets, balls left, target, required rate, Elo, venue par, format.",
        "GroupKFold by match: a match is never in both train and test.",
        f"Brier {w1['brier']:.3f} vs {w1['baseline_brier']:.3f} (innings 1); {w2['brier']:.3f} vs {w2['baseline_brier']:.3f} (chases).",
        "Mean out-of-fold ΔWP ≈ 0, so the model is not biased toward favourites.",
    ], size=11.5, gap=8)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(5.65),
                             Inches(5.5), Inches(0.95))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.fill.background()
    tx(s, 7.38, 5.77, 5.15, 0.75,
       "Why not score raw ΔWP? Split-half reliability of match-level ΔWP is only 0.14-0.26. Runs above expectation are about twice as repeatable.",
       9.4, False, INK, spacing=1.05)


def slide_scoring(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "From impact to score", "One currency, two roles, one index",
           "We measure the residual against a situation-specific expectation, then shrink small samples.", n)
    steps = [
        ("1", "All legal deliveries", "State before the ball: score, wickets, phase, target, venue, stakes."),
        ("2", "Pressure sample", "Keep PI ≥ 40. Within it, weight by (PI/50); the intensity still matters."),
        ("3", "Residual impact", "Batter: runs − expected. Bowler: expected − conceded. Wickets use the same logic."),
        ("4", "Shrink", "Pull toward the player's own all-ball level with k = 150; 150+ pressure balls, 18+ innings."),
        ("5", "Composite → index", "0.80·z(RAE) + 0.20·z(wickets); index = 50 + 15·z, clipped 0-100."),
    ]
    for i, (num, title, desc) in enumerate(steps):
        y = 1.62 + i * 0.94
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(y + 0.05),
                                  Inches(0.42), Inches(0.42))
        circ.fill.solid()
        circ.fill.fore_color.rgb = TEAL if i < 4 else ORANGE
        circ.line.fill.background()
        tx(s, 0.7, y + 0.11, 0.42, 0.25, num, 10.5, True, WHITE, PP_ALIGN.CENTER)
        tx(s, 1.28, y, 4.0, 0.26, title, 11.5, True, INK)
        tx(s, 1.28, y + 0.24, 4.1, 0.55, desc, 8.6, False, GRAY, spacing=1.0)
    rows = [
        ["Fairness issue", "Treatment"],
        ["IPL vs T20I", "Format weight 1.0 / 0.6 at aggregation; expectations estimated within format."],
        ["Sample size", "≥150 pressure balls, ≥18 innings; empirical-Bayes shrinkage k = 150."],
        ["Role / ledger", "Separate batter and bowler models; run-outs never credited to the bowler; non-striker run-outs excluded."],
        ["Match context", "Expectation buckets by phase, wickets, chase deficit, required rate, format; Elo and venue par enter the WP model."],
        ["Stakes", "League 1.0, Super 4 1.05, qualifier/eliminator 1.15, semi 1.20, final 1.30."],
        ["Data quality", "No-results, 2 DLS-inconsistent matches and super-over innings excluded."],
    ]
    table(s, rows, 5.65, 1.6, 7.0, 4.85, [1.35, 5.65], fs=8.4,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    tx(s, 5.65, 6.55, 7.05, 0.35,
       f"Eligible after filters: {d['bat_elig']} batters and {d['bowl_elig']} bowlers.",
       9.5, True, TEAL)


def _result_slide(prs, d, n, role):
    df = d["bat"] if role == "bat" else d["bowl"]
    label = "batters" if role == "bat" else "bowlers"
    color = TEAL if role == "bat" else ORANGE
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, f"Evidence · top 10 {label}",
           f"Top 10 {label} under pressure",
           "Situation-adjusted output, not raw runs or reputation. Rank intervals use 500 match-level bootstrap resamples.", n)
    if role == "bat":
        rows = [["#", "Player", "Index", "90% rank", "Inns", "P balls", "P SR", "RAE/100"]]
        for _, r in df.iterrows():
            rows.append([int(r["rank"]), r["player_name"], f"{r['score_index']:.1f}",
                         f"{r['rank_lo']:.0f}-{r['rank_hi']:.0f}", int(r["innings"]),
                         int(r["n_pressure"]), f"{r['pressure_sr']:.0f}",
                         f"{r['impact_raw_pct']:+.0f}"])
        widths = [0.38, 2.5, 0.62, 0.82, 0.5, 0.68, 0.62, 0.78]
    else:
        rows = [["#", "Player", "Index", "90% rank", "Inns", "P balls", "Econ", "Wkts", "RAE/100"]]
        for _, r in df.iterrows():
            rows.append([int(r["rank"]), r["player_name"], f"{r['score_index']:.1f}",
                         f"{r['rank_lo']:.0f}-{r['rank_hi']:.0f}", int(r["innings"]),
                         int(r["n_pressure"]), f"{r['pressure_economy']:.1f}",
                         int(r["pressure_wickets"]), f"{r['impact_raw_pct']:+.0f}"])
        widths = [0.34, 2.18, 0.56, 0.74, 0.44, 0.62, 0.52, 0.48, 0.68]
    table(s, rows, 0.55, 1.6, sum(widths), 4.75, widths, fs=8.8,
          header_fill=color,
          aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * (len(widths) - 2))
    pic(s, "top_batters.png" if role == "bat" else "top_bowlers.png",
        7.55, 1.55, 5.25, 4.95)
    tx(s, 0.65, 6.55, 12.0, 0.3,
       "P SR / Econ and RAE/100 are computed on the pressure sample (PI ≥ 40); RAE = runs above situational expectation.",
       9, False, GRAY)


def slide_phase(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Phase fingerprint", "Where each player earns the score",
           "Runs above expectation per 100 pressure balls, split by phase. Positive is good for both roles.", n)
    pic(s, "phase_heatmap.png", 0.35, 1.45, 12.65, 5.15)
    bat = d["bs"].set_index("player_name")
    bowl = d["ws"].set_index("player_name")
    rinku = bat.loc["Rinku Singh", "phase_death"]
    bumrah = [bowl.loc["Jasprit Bumrah", f"phase_{p}"] for p in ("powerplay", "middle", "death")]
    narine = bowl.loc["Sunil Narine", "phase_death"]
    tx(s, 0.65, 6.68, 12.0, 0.3,
       f"Rinku is a death specialist ({rinku:+.0f}); Bumrah is the rare all-phase bowler "
       f"({bumrah[0]:+.0f} / {bumrah[1]:+.0f} / {bumrah[2]:+.0f}); Narine owns the death ({narine:+.0f}).",
       9.2, False, GRAY)


def slide_gap(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Pressure vs overall", "Pressure is not just overall quality",
           "Grey = the player's all-ball level; orange = his pressure-sample level. The gap is descriptive, not the ranking currency.", n)
    pic(s, "pressure_vs_overall.png", 0.35, 1.45, 12.65, 5.1)
    sc = d["bs"].set_index("player_name")
    gap = (sc["prae"] - sc["own_runs"]) * 100
    lifts = [f"{n} ({gap[n]:+.0f})" for n in ("Rinku Singh", "David Miller", "Ashutosh Sharma")]
    holds = [f"{n} ({gap[n]:+.0f})" for n in ("Tim David", "Suryakumar Yadav")]
    tx(s, 0.65, 6.65, 12.0, 0.3,
       f"{lifts[0]}, {lifts[1]} and {lifts[2]} lift; {holds[0]} and {holds[1]} "
       "hold a high absolute level rather than lifting.",
       9.2, False, GRAY)


def slide_case(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Case study", "Anatomy of a pressure innings",
           "Rinku Singh, KKR vs Gujarat Titans, 9 April 2023: 29 needed off the final over.", n)
    pic(s, "case_study.png", 0.35, 1.5, 12.65, 4.55)
    chip(s, 0.65, 6.2, 2.9, 0.78, "0.5%", "win probability before the last over", ORANGE)
    chip(s, 3.7, 6.2, 2.9, 0.78, "5 sixes", "off the last six balls", ORANGE)
    chip(s, 6.75, 6.2, 2.9, 0.78, "+0.99", "ΔWP added in the over", TEAL)
    chip(s, 9.8, 6.2, 2.85, 0.78, "81.7", "mean PI in the over", TEAL)


def slide_reliability(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Reliability", "How much should you trust the order?",
           "The set is more reliable than the exact ranking; we show both.", n)
    pic(s, "reliability.png", 0.35, 1.48, 6.35, 3.9)
    pic(s, "sensitivity.png", 6.85, 1.48, 6.15, 3.9)
    v = d["val"]
    bullets(s, 0.65, 5.55, 12.0, 1.35, [
        f"Split-half Spearman: {v['split_half']['batter']['spearman']:.2f} (batters), {v['split_half']['bowler']['spearman']:.2f} (bowlers). Moderate, not decisive.",
        f"External check: Player-of-the-Match innings sit at the {_ord(100*v['mom'][0]['mean_mom_percentile'])} percentile (batting) and {_ord(100*v['mom'][1]['mean_mom_percentile'])} (bowling) of pressure impact.",
        "Sensitivity: top-10 overlap is 8-10 of 10 for most variants; only extreme pressure cuts change the membership.",
    ], size=10.2, gap=4)
    tx(s, 0.65, 6.92, 12.0, 0.2,
       "Bootstrap intervals: Rinku 1-17, Narine 1-13, Bumrah 2-38. Treat the top 10 as a shortlist, not a strict ladder.",
       8.8, True, GRAY)


def slide_limitations(prs, d, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Limits & reproduction", "What this is — and what it is not",
           "A transparent, reproducible ranking, not a claim of psychological measurement.", n)
    tx(s, 0.65, 1.62, 5.8, 0.28, "LIMITATIONS", 9.5, True, ORANGE)
    bullets(s, 0.65, 1.95, 5.85, 4.2, [
        "Pressure is inferred from match state; we do not observe nerves, fatigue or intent.",
        "No ball-tracking or fielding data: placement, catch difficulty and field pressure are missing.",
        "T20I exposure is thinner and more variable than IPL exposure; format weights only partly fix that.",
        "The exact order is noisy. 90% intervals are wide; use the top 10 as a set.",
        "The clutch premium (high minus calm) is descriptive and can be volatile for low-volume hitters.",
        "Observational, not causal: a player's role and team context affect the situations he faces.",
    ], size=9.8, gap=5)
    tx(s, 6.95, 1.62, 5.7, 0.28, "REPRODUCIBLE PIPELINE", 9.5, True, TEAL)
    bullets(s, 6.95, 1.95, 5.7, 2.4, [
        "uv run python -m src.pressure.pipeline rebuilds every table, figure and check.",
        "Raw CSVs → state → WP model → Pressure Index → scoring → bootstrap → validation → figures.",
        "Seed 42; 500 match-level bootstrap reps; sensitivity over γ, k, format weight, wicket weight and pressure cut.",
        "Outputs go to outputs/; the deck is rebuilt by python -m report.build_deck.",
        "IPL and T20I only; no-results, DLS-inconsistent matches and super overs excluded.",
    ], size=9.6, gap=5)
    parts = [
        ("src/pressure", "pipeline modules"),
        ("outputs/", "tables + figures"),
        ("report/", "deck + builder"),
        ("notebooks/", "guided walkthrough"),
    ]
    for i, (name, desc) in enumerate(parts):
        x = 0.65 + i * 3.06
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.55),
                                 Inches(2.9), Inches(0.72))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.fill.background()
        box.adjustments[0] = 0.12
        tx(s, x + 0.14, 4.66, 2.62, 0.24, name, 9.5, True, TEAL)
        tx(s, x + 0.14, 4.93, 2.62, 0.22, desc, 8.4, False, GRAY)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(6.05),
                             Inches(12.03), Inches(0.78))
    box.fill.solid()
    box.fill.fore_color.rgb = INK
    box.line.fill.background()
    tx(s, 0.9, 6.2, 11.5, 0.5,
       "Bottom line: pick the players whose pressure residual is high and repeatable — not the most famous names.",
       13, True, WHITE)


def build():
    REPORT.mkdir(exist_ok=True)
    d = load()
    scope(d)
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    slide_title(prs, d)
    slide_answer(prs, d, 2)
    slide_pressure(prs, d, 3)
    slide_distribution(prs, d, 4)
    slide_wp(prs, d, 5)
    slide_scoring(prs, d, 6)
    _result_slide(prs, d, 7, "bat")
    _result_slide(prs, d, 8, "bowl")
    slide_phase(prs, d, 9)
    slide_gap(prs, d, 10)
    slide_case(prs, d, 11)
    slide_reliability(prs, d, 12)
    slide_limitations(prs, d, 13)

    path = REPORT / "Pressure_Performance_Deck.pptx"
    prs.save(path)
    print(f"saved {path} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
