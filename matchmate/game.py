"""Over predictions: a free-to-play mini-game played between overs.

Before each over the fan answers up to three yes/no questions. Each question shows the
historical rate for overs like this one (same over number, similar wickets down), and a
correct pick pays more when it was the less likely outcome, like odds.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache

from .config import LEADERBOARD_PATH, OVER_RATES_PATH
from .replay import Replay

QUESTIONS = {
    "runs10": "Will this over go for 10+ runs?",
    "wicket": "Will a wicket fall this over?",
    "six": "Will there be a six this over?",
}
BOT_FANS = ["CoverDriveKaran", "YorkerQueen", "DeathOverDev", "SpinWizard42", "PowerplayPriya"]


@lru_cache(maxsize=1)
def base_rates() -> dict:
    return json.loads(OVER_RATES_PATH.read_text())


def p_yes(question: str, over: int, wkts_down: int) -> float:
    wb = "0-2" if wkts_down <= 2 else ("3-5" if wkts_down <= 5 else "6+")
    rates = base_rates()
    bucket = rates.get(f"{over}|{wb}") or rates["_overall"]
    return float(bucket[question])


def points_for(p_pick: float) -> int:
    """5 / probability of the pick, so a 50/50 call is worth 10 and a 20% call 25."""
    return int(round(5 / max(p_pick, 0.05)))


@dataclass
class OverRef:
    innings: int
    over: int  # 0-based

    @property
    def key(self) -> str:
        return f"{self.innings}-{self.over}"

    @property
    def label(self) -> str:
        return f"Innings {self.innings}, over {self.over + 1}"


def upcoming_over(replay: Replay, idx: int) -> OverRef | None:
    """The over about to start, if the replay sits between overs. None mid-over."""
    if replay.finished(idx):
        return None
    r = replay.row(idx)
    nxt = replay.row(idx + 1)
    if r.is_start:
        return OverRef(int(r.innings), 0)
    if nxt.is_start:
        return OverRef(int(nxt.innings), 0)
    if int(r.ball) >= 6 and nxt.over != r.over:
        return OverRef(int(nxt.innings), int(nxt.over))
    return None


def over_result(replay: Replay, ref: OverRef, idx: int) -> dict | None:
    """What happened in the over, once the replay has moved past its last ball."""
    tl = replay.tl
    g = tl[(tl.innings == ref.innings) & (tl.over == ref.over) & ~tl.is_start]
    if g.empty or g.index.max() > idx:
        return None  # over not finished yet (or never bowled)
    return {"runs10": bool(g.runs_total.sum() >= 10), "wicket": bool(g.is_wicket.any()),
            "six": bool((g.runs_batter == 6).any()), "runs": int(g.runs_total.sum())}


def innings_ended_before(replay: Replay, ref: OverRef, idx: int) -> bool:
    """The over was never bowled (innings/match ended) and the replay is past that point."""
    tl = replay.tl
    g = tl[(tl.innings == ref.innings) & (tl.over == ref.over) & ~tl.is_start]
    if not g.empty:
        return False
    inn_rows = tl[tl.innings == ref.innings]
    return inn_rows.empty or inn_rows.index.max() <= idx


def settle(picks: dict, replay: Replay, idx: int) -> dict:
    """Score any unsettled picks whose over has finished. `picks` maps over key ->
    {question: {"pick": bool, "p_yes": float}}. Mutates and returns a summary."""
    total = correct = answered = 0
    for key, qs in picks.items():
        inn, ov = map(int, key.split("-"))
        ref = OverRef(inn, ov)
        res = over_result(replay, ref, idx)
        void = res is None and innings_ended_before(replay, ref, idx)
        for q, p in qs.items():
            if p.get("settled"):
                pass
            elif res is not None:
                p["settled"] = True
                p["actual"] = res[q]
                p["correct"] = p["pick"] == res[q]
                prob = p["p_yes"] if p["pick"] else 1 - p["p_yes"]
                p["points"] = points_for(prob) if p["correct"] else 0
                p["over_runs"] = res["runs"]
            elif void:
                p.update(settled=True, void=True, correct=False, points=0, actual=None)
            if p.get("settled") and not p.get("void"):
                answered += 1
                correct += int(p["correct"])
                total += p["points"]
    return {"points": total, "correct": correct, "answered": answered}


def bot_scores(replay: Replay, idx: int) -> dict[str, int]:
    """Simulated fans playing the same match, for a match leaderboard. Each bot answers
    ~60% of overs, picking by the historical rate plus noise, with a fixed seed."""
    out = {}
    tl = replay.tl
    overs = tl[~tl.is_start].groupby(["innings", "over"]).agg(
        wb=("wkts", "first"))
    for b_i, bot in enumerate(BOT_FANS):
        rng = random.Random(f"{replay.match_id}-{bot}")
        pts = 0
        skill = 0.04 * b_i
        for (inn, ov), row in overs.iterrows():
            if rng.random() > 0.6:
                continue
            ref = OverRef(int(inn), int(ov))
            res = over_result(replay, ref, idx)
            if res is None:
                continue
            for q in QUESTIONS:
                if rng.random() > 0.7:
                    continue
                py = p_yes(q, int(ov), int(row.wb))
                pick = rng.random() < (py + skill * (1 if res[q] else -1))
                if pick == res[q]:
                    pts += points_for(py if pick else 1 - py)
        out[bot] = pts
    return out


# ------------------------------------------------------------- persistent hall of fame
def load_hall_of_fame() -> list[dict]:
    try:
        return json.loads(LEADERBOARD_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_score(name: str, match_label: str, points: int, correct: int, answered: int) -> None:
    board = load_hall_of_fame()
    board.append({"name": name[:24], "match": match_label, "points": points,
                  "correct": correct, "answered": answered})
    board = sorted(board, key=lambda e: -e["points"])[:50]
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(json.dumps(board, indent=1))
