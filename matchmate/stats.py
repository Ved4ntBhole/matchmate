"""Player, venue and head-to-head stats from Cricsheet IPL data.

All functions take `before` (the replayed match's date) so stats only use matches
played before it: what a fan could have known on the day.
"""
from __future__ import annotations

import difflib
from functools import lru_cache

import pandas as pd

from .replay import load_deliveries, load_matches

PHASES = {"powerplay": (0, 5), "middle": (6, 14), "death": (15, 19)}


@lru_cache(maxsize=1)
def all_players() -> list[str]:
    d = load_deliveries()
    return sorted(set(d.batter) | set(d.bowler) | set(d.non_striker))


def resolve_player(query: str, prefer: list[str] | None = None) -> str | None:
    """Match 'Bumrah', 'jasprit bumrah' or 'JJ Bumrah' to Cricsheet's 'JJ Bumrah'.
    `prefer` (players in the current match) wins ties."""
    q = query.strip().lower().replace(".", " ")
    if not q:
        return None
    pool = all_players()
    tokens = q.split()
    surname, first = tokens[-1], (tokens[0] if len(tokens) > 1 else None)
    for group in (list(prefer or []), pool):
        exact = [p for p in group if p.lower() == q]
        if exact:
            return exact[0]
        # Surname match, e.g. "bumrah" or "jasprit bumrah" -> "JJ Bumrah".
        hits = [p for p in group if surname in p.lower().split()]
        if first and len(hits) > 1:
            # "Jasprit" -> initial J must appear among the leading initials/names.
            narrowed = [p for p in hits if any(w.lower().startswith(first[0])
                                               for w in p.split()[:-1])]
            hits = narrowed or hits
        if len(hits) == 1:
            return hits[0]
        if hits:
            d = load_deliveries()
            counts = {h: int((d.batter == h).sum() + (d.bowler == h).sum()) for h in hits}
            return max(counts, key=counts.get)
    close = difflib.get_close_matches(query, pool, n=1, cutoff=0.6)
    return close[0] if close else None


def _prior(before) -> pd.DataFrame:
    d = load_deliveries()
    return d[d.date < pd.Timestamp(before)] if before is not None else d


def _phase_filter(d: pd.DataFrame, phase: str) -> pd.DataFrame:
    if phase in PHASES:
        lo, hi = PHASES[phase]
        return d[(d.over >= lo) & (d.over <= hi)]
    return d


def batting_stats(player: str, before=None, phase: str = "all", d: pd.DataFrame | None = None) -> dict:
    d = _phase_filter(_prior(before) if d is None else d, phase)
    b = d[d.batter == player]
    faced = b[b.extra_type != "wides"]
    runs = int(b.runs_batter.sum())
    balls = len(faced)
    outs = int((d.player_out == player).sum())
    inns = b.groupby("match_id").runs_batter.sum()
    return {
        "innings": int(b.match_id.nunique()), "runs": runs, "balls": balls,
        "dismissals": outs,
        "average": round(runs / outs, 1) if outs else None,
        "strike_rate": round(runs * 100 / balls, 1) if balls else None,
        "fours": int((faced.runs_batter == 4).sum()), "sixes": int((faced.runs_batter == 6).sum()),
        "dot_ball_pct": round(100 * (faced.runs_batter == 0).mean(), 1) if balls else None,
        "fifties": int(((inns >= 50) & (inns < 100)).sum()), "hundreds": int((inns >= 100).sum()),
        "highest": int(inns.max()) if len(inns) else 0,
    }


def bowling_stats(player: str, before=None, phase: str = "all", d: pd.DataFrame | None = None) -> dict:
    d = _phase_filter(_prior(before) if d is None else d, phase)
    b = d[d.bowler == player]
    legal = int(b.legal.sum())
    conceded = int((b.runs_batter + b.runs_extras.where(
        b.extra_type.isin(["wides", "noballs"]), 0)).sum())
    wkts = int(b.bowler_wicket.sum())
    return {
        "matches": int(b.match_id.nunique()), "overs": f"{legal // 6}.{legal % 6}",
        "runs_conceded": conceded, "wickets": wkts,
        "economy": round(conceded * 6 / legal, 2) if legal else None,
        "average": round(conceded / wkts, 1) if wkts else None,
        "strike_rate": round(legal / wkts, 1) if wkts else None,
        "dot_ball_pct": round(100 * ((b.runs_total == 0) & b.legal).sum() / legal, 1) if legal else None,
        "sixes_conceded": int((b.runs_batter == 6).sum()),
    }


def head_to_head(batter: str, bowler: str, before=None) -> dict:
    d = _prior(before)
    h = d[(d.batter == batter) & (d.bowler == bowler)]
    faced = h[h.extra_type != "wides"]
    runs = int(h.runs_batter.sum())
    outs = int(((h.player_out == batter) & h.bowler_wicket).sum())
    return {"batter": batter, "bowler": bowler, "balls": len(faced), "runs": runs,
            "dismissals": outs,
            "strike_rate": round(runs * 100 / len(faced), 1) if len(faced) else None,
            "fours": int((faced.runs_batter == 4).sum()), "sixes": int((faced.runs_batter == 6).sum())}


def venue_stats(venue: str, before=None) -> dict:
    m = load_matches()
    m = m[(m.venue == venue) & m.winner.notna()]
    if before is not None:
        m = m[m.date < pd.Timestamp(before)]
    d = _prior(before)
    first = (d[(d.venue == venue) & (d.innings == 1)]
             .groupby("match_id").agg(date=("date", "first"), runs=("inn_runs", "last"))
             .sort_values("date")["runs"])  # chronological, for the "last 10" figure
    chase_wins = (m.winner == m.team2).sum()
    return {"venue": venue, "matches": int(len(m)),
            "avg_first_innings": round(float(first.mean()), 1) if len(first) else None,
            "recent_avg_first_innings_last_10": round(float(first.tail(10).mean()), 1) if len(first) else None,
            "chasing_win_pct": round(100 * chase_wins / len(m), 1) if len(m) else None,
            "highest_first_innings": int(first.max()) if len(first) else None}


def team_head_to_head(team_a: str, team_b: str, before=None) -> dict:
    m = load_matches()
    m = m[((m.team1 == team_a) & (m.team2 == team_b)) | ((m.team1 == team_b) & (m.team2 == team_a))]
    if before is not None:
        m = m[m.date < pd.Timestamp(before)]
    return {"matches": int(len(m)), f"{team_a}_wins": int((m.winner == team_a).sum()),
            f"{team_b}_wins": int((m.winner == team_b).sum()),
            "last_5_winners": m.sort_values("date").winner.tail(5).fillna("No result").tolist()}


def latest_team(player: str, before=None) -> str | None:
    """The team a player last appeared for in IPL matches before `before`."""
    d = _prior(before)
    rows = d[(d.batter == player) | (d.non_striker == player) | (d.bowler == player)]
    if rows.empty:
        return None
    last = rows.iloc[-1]
    return last.bowling_team if last.bowler == player else last.batting_team
