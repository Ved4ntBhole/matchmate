"""Match-replay engine: plays a finished match back ball by ball as if it were live.

A replay is just a pointer (`idx`) into the match's precomputed timeline. Everything
the app, the agent and the game see is computed from rows [0..idx], so nothing from
the future of the match can leak out (no spoilers, no lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .config import DELIVERIES_PATH, MATCHES_PATH, TIMELINES_PATH, team_code

# Alert thresholds by innings. First-innings probabilities move less (the target is
# still unknown), so smaller swings there are already big moments.
SWING_BALL = {1: 0.06, 2: 0.10}   # one ball moving win prob this much is an alert
SWING_OVER = {1: 0.10, 2: 0.15}   # a whole over moving it this much is an alert


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PATH)


@lru_cache(maxsize=1)
def load_timelines() -> pd.DataFrame:
    return pd.read_parquet(TIMELINES_PATH)


@lru_cache(maxsize=1)
def load_deliveries() -> pd.DataFrame:
    d = pd.read_parquet(DELIVERIES_PATH)
    m = load_matches().set_index("match_id")
    d["date"] = d["match_id"].map(m["date"])
    d["venue"] = d["match_id"].map(m["venue"])
    # Chronological order, so "most recent" / "last N matches" stats are correct
    # (Cricsheet match ids are not in date order).
    d = d.sort_values(["date", "match_id", "seq"], kind="stable").reset_index(drop=True)
    return d


def overs_str(legal_balls: int) -> str:
    return f"{int(legal_balls) // 6}.{int(legal_balls) % 6}"


def ball_outcome(row) -> str:
    if row.is_start:
        return ""
    if row.is_wicket:
        return "WICKET"
    if row.extra_type == "wides":
        return f"wide (+{int(row.runs_total)})"
    if row.extra_type == "noballs":
        return f"no ball (+{int(row.runs_total)})"
    if row.runs_batter == 6:
        return "SIX"
    if row.runs_batter == 4:
        return "FOUR"
    runs = int(row.runs_total)
    if runs == 0:
        return "dot ball"
    return f"{runs} run{'s' if runs > 1 else ''}"


def ball_text(row) -> str:
    if row.is_start:
        return (f"Innings {row.innings}: {row.batting_team} begin"
                + (f", chasing {int(row.target)}" if row.innings == 2 else ""))
    ov = f"{int(row.over)}.{max(int(row.ball), 1) if row.legal else int(row.ball) + 1}"
    text = f"{ov} {row.bowler} to {row.batter}: {ball_outcome(row)}"
    if row.is_wicket and row.player_out:
        text += f" ({row.player_out}, {row.wicket_kind})"
    return text


def ball_chip(row) -> str:
    """Short symbol for an over strip: 0 1 4 6 W wd nb."""
    if row.is_wicket:
        return "W"
    if row.extra_type == "wides":
        return "wd"
    if row.extra_type == "noballs":
        return "nb"
    if row.extra_type in ("byes", "legbyes"):
        return f"{int(row.runs_total)}{'b' if row.extra_type == 'byes' else 'lb'}"
    return str(int(row.runs_batter))


@dataclass
class Replay:
    match_id: str

    def __post_init__(self):
        tl = load_timelines()
        self.tl = tl[tl.match_id == self.match_id].reset_index(drop=True)
        self.meta = load_matches().set_index("match_id").loc[self.match_id]
        self.n = len(self.tl)
        self.team1, self.team2 = self.meta.team1, self.meta.team2
        self.code1, self.code2 = team_code(self.team1), team_code(self.team2)

    # ------------------------------------------------------------------ navigation
    @property
    def last(self) -> int:
        return self.n - 1

    def next_over_idx(self, idx: int) -> int:
        """Index of the last ball of the current over (or of the next over if idx
        already sits at an over end), so 'next over' always shows a full over."""
        tl = self.tl
        i = idx + 1
        if i > self.last:
            return self.last
        inn, over = tl.at[i, "innings"], tl.at[i, "over"]
        if tl.at[i, "is_start"]:
            # Step from innings start through its first over.
            over = tl.at[min(i + 1, self.last), "over"]
            i += 1
        j = i
        while j + 1 <= self.last and tl.at[j + 1, "innings"] == inn \
                and tl.at[j + 1, "over"] == over and not tl.at[j + 1, "is_start"]:
            j += 1
        return j

    def finished(self, idx: int) -> bool:
        return idx >= self.last

    # ------------------------------------------------------------------ state
    def row(self, idx: int):
        return self.tl.iloc[idx]

    def p_team1(self, idx: int) -> float:
        return float(self.tl.at[idx, "p_team1"])

    def pre_match_p(self) -> float:
        return float(self.tl.at[0, "p_team1"])

    def phase(self, legal_balls: int) -> str:
        if legal_balls < 36:
            return "powerplay (overs 1-6)"
        if legal_balls < 90:
            return "middle overs (7-15)"
        return "death overs (16-20)"

    def batters_now(self, idx: int) -> list[dict]:
        r = self.row(idx)
        cur = self.tl.iloc[: idx + 1]
        cur = cur[(cur.innings == r.innings) & ~cur.is_start]
        if cur.empty:
            return []
        out = []
        last = cur.iloc[-1]
        names = [last.batter, last.non_striker]
        if last.is_wicket and last.player_out in names:
            names = [n for n in names if n != last.player_out]
        for name in names:
            faced = cur[(cur.batter == name) & (cur.extra_type != "wides")]
            out.append({"name": name, "runs": int(cur[cur.batter == name].runs_batter.sum()),
                        "balls": int(len(faced)),
                        "fours": int((faced.runs_batter == 4).sum()),
                        "sixes": int((faced.runs_batter == 6).sum())})
        return out

    def bowler_now(self, idx: int) -> dict | None:
        r = self.row(idx)
        cur = self.tl.iloc[: idx + 1]
        cur = cur[(cur.innings == r.innings) & ~cur.is_start]
        if cur.empty:
            return None
        name = cur.iloc[-1].bowler
        return self._bowling_line(cur[cur.bowler == name], name)

    @staticmethod
    def _bowling_line(b: pd.DataFrame, name: str) -> dict:
        conceded = b.runs_batter + b.runs_extras.where(b.extra_type.isin(["wides", "noballs"]), 0)
        legal = int(b.legal.sum())
        return {"name": name, "overs": overs_str(legal), "runs": int(conceded.sum()),
                "wickets": int(b.bowler_wicket.sum()),
                "economy": round(conceded.sum() * 6 / legal, 2) if legal else 0.0}

    def snapshot(self, idx: int) -> dict:
        r = self.row(idx)
        snap = {
            "match": f"{self.team1} vs {self.team2}",
            "season": int(self.meta.season), "venue": self.meta.venue,
            "innings": int(r.innings), "batting_team": r.batting_team,
            "bowling_team": r.bowling_team,
            "score": f"{int(r.runs)}/{int(r.wkts)}",
            "overs": overs_str(r.legal_balls),
            "max_overs": overs_str(r.max_balls),
            "run_rate": round(r.runs * 6 / r.legal_balls, 2) if r.legal_balls else 0.0,
            "phase": self.phase(int(r.legal_balls)),
            "par_first_innings_score": round(float(r.par)),
            "batters": self.batters_now(idx),
            "bowler": self.bowler_now(idx),
            "last_ball": ball_text(r),
            "win_probability": {self.team1: round(self.p_team1(idx), 3),
                                self.team2: round(1 - self.p_team1(idx), 3)},
            "match_finished": self.finished(idx),
        }
        if r.innings == 2:
            need = int(r.target - r.runs)
            balls_left = int(r.max_balls - r.legal_balls)
            snap.update({
                "target": int(r.target), "runs_needed": max(need, 0), "balls_left": balls_left,
                "required_run_rate": round(need * 6 / balls_left, 2) if balls_left > 0 else None,
                "first_innings": self.innings_total(1),
            })
        if self.finished(idx):
            snap["result"] = self.result_text()
        return snap

    def innings_total(self, inn: int) -> str:
        t = self.tl[(self.tl.innings == inn)]
        if t.empty:
            return ""
        r = t.iloc[-1]
        return f"{r.batting_team} {int(r.runs)}/{int(r.wkts)} ({overs_str(r.legal_balls)} ov)"

    def result_text(self) -> str:
        m = self.meta
        if not m.winner:
            return "No result"
        return f"{m.winner} won {m.margin}".strip()

    # ------------------------------------------------------------------ swings
    def swings(self, idx: int) -> list[dict]:
        """Plain-language alerts for big win-probability swings up to idx."""
        tl = self.tl.iloc[: idx + 1]
        alerts = []
        p = tl.p_team1.to_numpy()
        for i in range(1, len(tl)):
            r = tl.iloc[i]
            if r.is_start:
                continue
            bat_is_1 = r.batting_team == self.team1
            before = p[i - 1] if bat_is_1 else 1 - p[i - 1]
            after = p[i] if bat_is_1 else 1 - p[i]
            if abs(after - before) >= SWING_BALL[int(r.innings)]:
                alerts.append(self._alert(i, r, before, after, "ball"))
        # Over-level swings that no single ball explains.
        for (inn, ov), g in tl[~tl.is_start].groupby(["innings", "over"], sort=False):
            i0, i1 = g.index[0], g.index[-1]
            if i0 == 0 or not self._over_complete(i1, idx):
                continue
            r = tl.iloc[i1]
            bat_is_1 = r.batting_team == self.team1
            before = p[i0 - 1] if bat_is_1 else 1 - p[i0 - 1]
            after = p[i1] if bat_is_1 else 1 - p[i1]
            if abs(after - before) >= SWING_OVER[int(inn)] and not any(
                    a["idx"] in g.index and a["kind"] == "ball" for a in alerts):
                alerts.append(self._alert(i1, r, before, after, "over", runs=int(g.runs_total.sum())))
        return sorted(alerts, key=lambda a: a["idx"])

    def _over_complete(self, i1: int, idx: int) -> bool:
        return i1 < idx or (i1 == idx and int(self.tl.at[i1, "ball"]) >= 6)

    def _alert(self, i, r, before, after, kind, runs=None) -> dict:
        code = team_code(r.batting_team)
        verb = "lifted" if after > before else "dropped"
        if kind == "over":
            what = f"That {runs}-run over"
        elif r.is_wicket:
            what = f"{r.player_out}'s wicket" if r.player_out else "That wicket"
        elif r.runs_batter == 6:
            what = f"{r.batter}'s six"
        elif r.runs_batter == 4:
            what = f"{r.batter}'s four"
        elif r.extra_type in ("wides", "noballs"):
            what = "That extra"
        elif r.runs_total == 0:
            what = "That dot ball"
        else:
            what = "That ball"
        text = f"{what} {verb} {code}'s chances from {before:.0%} to {after:.0%}."
        return {"idx": int(i), "kind": kind, "over": overs_str(r.legal_balls),
                "innings": int(r.innings), "team": r.batting_team, "before": float(before),
                "after": float(after), "delta": float(after - before), "text": text}

    # ------------------------------------------------------------------ scorecard
    def scorecard(self, idx: int, inn: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        cur = self.tl.iloc[: idx + 1]
        cur = cur[(cur.innings == inn) & ~cur.is_start]
        if cur.empty:
            return pd.DataFrame(), pd.DataFrame()
        order = list(dict.fromkeys(list(cur.batter) + list(cur.non_striker)))
        outs = cur[cur.is_wicket].set_index("player_out")["wicket_kind"].to_dict()
        bat_rows = []
        for name in order:
            b = cur[cur.batter == name]
            faced = b[b.extra_type != "wides"]
            balls = len(faced)
            runs = int(b.runs_batter.sum())
            bat_rows.append({"Batter": name, "R": runs, "B": balls,
                             "4s": int((faced.runs_batter == 4).sum()),
                             "6s": int((faced.runs_batter == 6).sum()),
                             "SR": round(runs * 100 / balls, 1) if balls else 0.0,
                             "Status": outs.get(name, "not out")})
        bowl_rows = []
        for name in dict.fromkeys(cur.bowler):
            line = self._bowling_line(cur[cur.bowler == name], name)
            bowl_rows.append({"Bowler": name, "O": line["overs"], "R": line["runs"],
                              "W": line["wickets"], "Econ": line["economy"]})
        return pd.DataFrame(bat_rows), pd.DataFrame(bowl_rows)

    def recent_balls(self, idx: int, n: int = 12) -> list[str]:
        cur = self.tl.iloc[max(0, idx - n + 1): idx + 1]
        return [ball_text(r) for r in cur.itertuples()]

    def current_over_chips(self, idx: int) -> list[str]:
        r = self.row(idx)
        if r.is_start:
            return []
        cur = self.tl.iloc[: idx + 1]
        g = cur[(cur.innings == r.innings) & (cur.over == r.over) & ~cur.is_start]
        return [ball_chip(x) for x in g.itertuples()]

    def over_runs(self, idx: int) -> pd.DataFrame:
        """Runs and wickets per completed/partial over, both innings, up to idx."""
        cur = self.tl.iloc[: idx + 1]
        cur = cur[~cur.is_start]
        g = cur.groupby(["innings", "over"]).agg(runs=("runs_total", "sum"),
                                                 wkts=("is_wicket", "sum"),
                                                 team=("batting_team", "first")).reset_index()
        g["over_no"] = g["over"] + 1
        return g
