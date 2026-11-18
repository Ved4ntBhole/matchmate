"""Match-state features shared by training and live inference.

The same `make_features` function builds the model input during training and while a
match is replayed, so the two can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_WINDOW = 30  # deliveries, roughly the last five overs
DEFAULT_PAR = 160.0
LEAGUE_WINDOW = 120  # recent first innings used for the league-wide par
VENUE_SHRINK = 8  # prior matches needed before a venue's own history gets half weight

# Innings 1 uses era-relative features only (projection vs par, not raw runs): raw
# totals drift upward season to season and hurt out-of-sample calibration.
INN1_FEATURES = ["wkts", "balls_left", "crr", "proj_minus_par", "runs_last_30",
                 "wkts_last_30"]
INN2_FEATURES = ["runs_needed", "balls_left", "wkts_left", "rrr", "crr", "rrr_minus_crr",
                 "target", "par", "needed_per_wkt", "runs_last_30", "wkts_last_30"]

# Direction each feature pushes the batting side's chances, used as XGBoost
# monotone constraints so the model can't learn nonsense like "more wickets = better".
INN1_MONOTONE = {"runs": 1, "wkts": -1, "balls_left": 0, "crr": 1, "par": -1,
                 "proj_minus_par": 1, "runs_last_30": 1, "wkts_last_30": -1}
INN2_MONOTONE = {"runs_needed": -1, "balls_left": 1, "wkts_left": 1, "rrr": -1, "crr": 0,
                 "rrr_minus_crr": -1, "target": 0, "par": 0, "needed_per_wkt": -1,
                 "runs_last_30": 0, "wkts_last_30": 0}


def add_par_scores(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Leakage-free par score for each match, using only matches played before it.

    par = recent league-wide first-innings average + a shrunk venue adjustment.
    The league part tracks scoring eras (e.g. the impact-player era from 2023).
    """
    inn1 = (deliveries[deliveries.innings == 1]
            .groupby("match_id").agg(score=("inn_runs", "last")))
    m = matches.sort_values(["date", "match_id"]).copy()
    m = m.join(inn1, on="match_id")
    qualifies = m["method"].isna() & (m["result"] != "no result") & (m["overs"] == 20)

    recent: list[float] = []
    venue_hist: dict[str, list[float]] = {}
    pars, leagues = [], []
    for _, row in m.iterrows():
        league = float(np.mean(recent[-LEAGUE_WINDOW:])) if recent else DEFAULT_PAR
        hist = venue_hist.get(row.venue, [])
        offset = (np.sum(hist) / (len(hist) + VENUE_SHRINK)) if hist else 0.0
        pars.append(league + offset)
        leagues.append(league)
        if qualifies.loc[row.name] and not np.isnan(row.score):
            recent.append(row.score)
            venue_hist.setdefault(row.venue, []).append(row.score - league)
    m["par"] = pars
    m["league_par"] = leagues
    return m.drop(columns=["score"])


def add_rolling(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    g = d.groupby(["match_id", "innings"])
    d["runs_last_30"] = g["runs_total"].transform(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).sum())
    d["wkts_last_30"] = g["is_wicket"].transform(
        lambda s: s.astype(int).rolling(ROLL_WINDOW, min_periods=1).sum())
    return d


def make_features(state: pd.DataFrame, innings: int) -> pd.DataFrame:
    """`state` needs: runs, wkts, legal_balls, max_balls, par, runs_last_30,
    wkts_last_30, and target (innings 2 only)."""
    s = state
    balls_left = (s["max_balls"] - s["legal_balls"]).clip(lower=0)
    overs_bowled = s["legal_balls"] / 6
    crr = np.where(s["legal_balls"] > 0, s["runs"] / overs_bowled.replace(0, np.nan), 0.0)
    crr = np.nan_to_num(crr)
    out = pd.DataFrame(index=s.index)
    out["balls_left"] = balls_left
    out["crr"] = crr
    out["par"] = s["par"]
    out["runs_last_30"] = s["runs_last_30"]
    out["wkts_last_30"] = s["wkts_last_30"]
    if innings == 1:
        out["runs"] = s["runs"]
        out["wkts"] = s["wkts"]
        # Projection blends the current rate with par rate, weighted by balls faced.
        par_rate = s["par"] / (s["max_balls"] / 6)
        w = s["legal_balls"] / s["max_balls"]
        blended = w * crr + (1 - w) * par_rate
        proj = s["runs"] + blended * balls_left / 6
        out["proj_minus_par"] = proj - s["par"] * s["max_balls"] / 120
        return out[INN1_FEATURES]
    runs_needed = (s["target"] - s["runs"]).clip(lower=0)
    wkts_left = (10 - s["wkts"]).clip(lower=0)
    rrr = np.where(balls_left > 0, runs_needed * 6 / balls_left.replace(0, np.nan), 36.0)
    rrr = np.clip(np.nan_to_num(rrr, nan=36.0), 0, 36)
    out["runs_needed"] = runs_needed
    out["wkts_left"] = wkts_left
    out["rrr"] = rrr
    out["rrr_minus_crr"] = rrr - crr
    out["target"] = s["target"]
    out["needed_per_wkt"] = runs_needed / wkts_left.clip(lower=1)
    return out[INN2_FEATURES]


def build_states(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery (plus one 'start of innings' row per innings) with the
    match state *after* that delivery, ready for make_features."""
    m = matches.set_index("match_id")
    d = add_rolling(deliveries)
    d = d.rename(columns={"inn_runs": "runs", "inn_wkts": "wkts"})

    starts = (d.groupby(["match_id", "innings"]).head(1)
              [["match_id", "innings", "batting_team", "bowling_team", "seq"]].copy())
    starts["seq"] = starts["seq"] - 0.5
    for col in ["runs", "wkts", "legal_balls", "runs_last_30", "wkts_last_30"]:
        starts[col] = 0
    starts["is_start"] = True
    d["is_start"] = False
    states = pd.concat([d, starts], ignore_index=True).sort_values(["match_id", "seq"])

    states["par"] = states["match_id"].map(m["par"])
    t_runs = states["match_id"].map(m["target_runs"])
    t_overs = states["match_id"].map(m["target_overs"]).fillna(states["match_id"].map(m["overs"]))
    states["target"] = np.where(states["innings"] == 2, t_runs, np.nan)
    states["max_balls"] = np.where(states["innings"] == 2, t_overs * 6,
                                   states["match_id"].map(m["overs"]).fillna(20) * 6)
    return states.reset_index(drop=True)
