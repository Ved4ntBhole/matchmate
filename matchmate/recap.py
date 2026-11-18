"""Match recap: five lines on the moments that decided the game.

Key facts (result, top performers, biggest win-probability swings) are computed from
data first; Gemini only turns them into prose, and a template does it offline.
"""
from __future__ import annotations

import json

from .agent import GeminiError, generate, get_api_key
from .config import team_code
from .replay import Replay


def _over_words(overs: str) -> str:
    """'15.1' -> 'ball 15.1 (the 16th over)'; '18.0' -> 'end of the 18th over'."""
    whole, ball = (int(x) for x in overs.split("."))
    n = whole + 1 if ball else whole
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"ball {overs} (the {n}{suffix} over)" if ball else f"end of the {n}{suffix} over"


def key_facts(replay: Replay, idx: int) -> dict:
    facts = {"match": f"{replay.team1} vs {replay.team2}", "venue": replay.meta.venue,
             "season": int(replay.meta.season), "finished": replay.finished(idx),
             "innings": []}
    for inn in (1, 2):
        bat, bowl = replay.scorecard(idx, inn)
        if bat.empty:
            continue
        top_bat = bat.sort_values(["R", "B"], ascending=[False, True]).head(2)
        top_bowl = bowl.sort_values(["W", "Econ"], ascending=[False, True]).head(2)
        rows = replay.tl.iloc[: idx + 1]
        last = rows[rows.innings == inn].iloc[-1]
        facts["innings"].append({
            "team": last.batting_team,
            "total": f"{int(last.runs)}/{int(last.wkts)} in {int(last.legal_balls) // 6}."
                     f"{int(last.legal_balls) % 6} overs",
            "top_batters": [f"{r.Batter} {r.R} ({r.B})" for r in top_bat.itertuples()],
            "top_bowlers": [f"{r.Bowler} {r.W}/{r.R} ({r.O} ov)" for r in top_bowl.itertuples()],
        })
    swings = sorted(replay.swings(idx), key=lambda a: -abs(a["delta"]))[:3]
    # "15.1" is ambiguous to an LLM (it once called it "the final over"), so spell it out.
    facts["turning_points"] = [f"{_over_words(a['over'])} of innings {a['innings']}: {a['text']}"
                               for a in swings]
    p0 = replay.pre_match_p()
    facts["pre_match_win_prob"] = {replay.team1: f"{p0:.0%}", replay.team2: f"{1 - p0:.0%}"}
    if replay.finished(idx):
        facts["result"] = replay.result_text()
        facts["player_of_match"] = replay.meta.player_of_match
        # Lowest point the eventual winner reached: a measure of how big the comeback was.
        if replay.meta.winner:
            p = replay.tl.p_team1.iloc[: idx + 1]
            pw = p if replay.meta.winner == replay.team1 else 1 - p
            facts["winner_lowest_win_prob"] = f"{pw.min():.0%}"
    else:
        p = replay.p_team1(idx)
        facts["current_win_prob"] = {replay.team1: f"{p:.0%}", replay.team2: f"{1 - p:.0%}"}
    return facts


def template_recap(f: dict) -> list[str]:
    lines = []
    if f.get("result"):
        lines.append(f"🏆 {f['result']} at {f['venue']}.")
    else:
        lines.append(f"⏱️ So far at {f['venue']}: " + ", ".join(
            f"{k} {v}" for k, v in f["current_win_prob"].items()) + " to win.")
    for i, inn in enumerate(f["innings"]):
        live = not f["finished"] and i == len(f["innings"]) - 1
        lines.append(f"🏏 {team_code(inn['team'])} {'are' if live else 'made'} {inn['total']}, led by "
                     + " and ".join(inn["top_batters"]) + ".")
    best_bowl = [b for inn in f["innings"] for b in inn["top_bowlers"][:1]]
    if best_bowl:
        lines.append("🎯 Best with the ball: " + "; ".join(best_bowl) + ".")
    if f["turning_points"]:
        lines.append("📈 Turning point: " + f["turning_points"][0].split(": ", 1)[1])
    if f.get("winner_lowest_win_prob"):
        lines.append(f"🔁 The winners' chances bottomed out at {f['winner_lowest_win_prob']}"
                     + (f"; player of the match: {f['player_of_match']}." if f.get("player_of_match") else "."))
    return lines[:5]


PROMPT = """Write a match recap for a casual cricket fan in exactly 5 lines.
Each line: one emoji, then one punchy sentence (max 25 words). Cover: the result (or the \
situation if unfinished), the standout batting, the standout bowling, the turning point \
(quote the win-probability swing), and one line on how the momentum went. Use only these \
facts; never invent numbers, teams or events, and describe overs exactly as written:

{facts}"""


def make_recap(replay: Replay, idx: int) -> dict:
    facts = key_facts(replay, idx)
    key = get_api_key()
    error = None
    if key:
        try:
            text, name = generate(key, PROMPT.format(facts=json.dumps(facts, indent=1)))
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 3:
                return {"lines": lines[:5], "facts": facts, "engine": name}
            error = "Gemini returned too few lines"
        except GeminiError as e:
            error = str(e)
    out = {"lines": template_recap(facts), "facts": facts, "engine": "template"}
    if error:
        out["error"] = error
    return out
