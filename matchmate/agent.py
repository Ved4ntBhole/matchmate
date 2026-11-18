"""'Ask the match': a Gemini agent whose answers are grounded in tool calls.

The model never sees raw data up front. It must call tools (score, win probability,
player stats...) that read only the replay up to the current ball, so it can't make up
numbers and can't spoil what happens next. If no Gemini key is configured, a small
rule-based fallback answers the common questions from the same tools.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from . import stats
from .replay import Replay
from .winprob import prob_batting_side

# The SDK logs an advisory about automatic function calling on every request.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Tried in order until one answers. Fast, widely available models first: on 2026-11-18
# gemini-3.8-flash was returning 503 "high demand" and gemini-flash-latest was 404 for
# new keys, while 2.5-flash / 3.1-flash-lite answered in 1-6 s. Pin one with GEMINI_MODEL.
DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash",
                  "gemini-3.8-flash"]

SYSTEM_PROMPT = """You are MatchMate, a friendly second-screen companion for a casual cricket fan \
watching a live IPL match.

Rules:
- Ground every number in a tool result. Never invent scores, stats or probabilities. If the \
tools can't answer, say so plainly.
- Call get_score() first for anything about the current match situation.
- The match is live: you only know what has happened up to the current ball. Never predict \
the result as a certainty.
- Write for a casual fan: 2-4 short sentences, plain English, explain jargon (e.g. "required \
rate = runs needed per over"). Use the team short codes or names the fan uses.
- Player career stats cover IPL matches before today's game only; say "in the IPL" not "career".
- Only state facts that appear in tool results. Don't add outside knowledge such as a \
player's team, role or reputation unless a tool returned it.
- Overs: "15.1" means the first ball of the 16th over. Say "the 16th over", not "over 15".
- When a question is about whether to worry or who is ahead, use get_win_prob() and quote \
the percentage, then give the one or two reasons that matter most.
"""


@dataclass
class MatchContext:
    replay: Replay
    idx: int
    tool_log: list[str] = field(default_factory=list)

    @property
    def date(self):
        return self.replay.meta.date

    def players_in_match(self) -> list[str]:
        tl = self.replay.tl.iloc[: self.idx + 1]
        return list(dict.fromkeys(list(tl.batter.dropna()) + list(tl.bowler.dropna())))

    def resolve(self, name: str) -> str | None:
        return stats.resolve_player(name, prefer=self.players_in_match())

    def team_of(self, name: str) -> str:
        """The player's team in this match, else their most recent IPL team before it.
        Given to the model so it never has to guess (it once said Chahal played for RR)."""
        tl = self.replay.tl.iloc[: self.idx + 1]
        bat = tl[(tl.batter == name) | (tl.non_striker == name)]
        if not bat.empty:
            return f"{bat.iloc[-1].batting_team} (in this match)"
        bowl = tl[tl.bowler == name]
        if not bowl.empty:
            return f"{bowl.iloc[-1].bowling_team} (in this match)"
        last = stats.latest_team(name, self.date)
        return f"{last} (most recent IPL team before today)" if last else "unknown"


def make_tools(ctx: MatchContext) -> list:
    """Tool functions bound to the live match context. Type hints and docstrings
    become the function-calling schema the model sees."""

    def _log(name, **kw):
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        ctx.tool_log.append(f"{name}({args})")

    def get_score() -> dict:
        """Current match situation: score, overs, run rates, target and runs needed (in a
        chase), batters at the crease, current bowler, last ball, and win probability."""
        _log("get_score")
        return ctx.replay.snapshot(ctx.idx)

    def get_win_prob() -> dict:
        """Win probability for each team now, how it changed over the last over and the
        last 5 overs, the pre-match probability, and the biggest swings so far."""
        _log("get_win_prob")
        r, rp = ctx.replay, ctx.replay.tl
        p_now = r.p_team1(ctx.idx)
        legal = rp.legal_balls.iloc[: ctx.idx + 1]
        inn = rp.innings.iloc[: ctx.idx + 1]

        def p_ago(balls):
            cur_inn, cur_lb = inn.iloc[-1], legal.iloc[-1]
            mask = (inn == cur_inn) & (legal <= max(cur_lb - balls, 0))
            j = mask[mask].index.max() if mask.any() else 0
            return r.p_team1(int(j))

        swings = sorted(r.swings(ctx.idx), key=lambda a: -abs(a["delta"]))[:3]
        return {
            "now": {r.team1: round(p_now, 3), r.team2: round(1 - p_now, 3)},
            "pre_match": {r.team1: round(r.pre_match_p(), 3), r.team2: round(1 - r.pre_match_p(), 3)},
            f"{r.team1}_change_last_over_pts": round(100 * (p_now - p_ago(6)), 1),
            f"{r.team1}_change_last_5_overs_pts": round(100 * (p_now - p_ago(30)), 1),
            "biggest_swings": [a["text"] + f" (over {a['over']}, innings {a['innings']})" for a in swings],
            "model": "logistic regression on runs needed, balls left, wickets, run rates and "
                     "venue par; trained on IPL 2008-2024",
        }

    def player_stats(player_name: str, phase: str = "all") -> dict:
        """IPL batting and bowling stats for a player from matches before today, plus what
        they have done in this match so far. phase: 'all', 'powerplay' (overs 1-6),
        'middle' (7-15) or 'death' (16-20)."""
        _log("player_stats", player_name=player_name, phase=phase)
        name = ctx.resolve(player_name)
        if not name:
            return {"error": f"No IPL player found matching '{player_name}'."}
        today = ctx.replay.tl.iloc[: ctx.idx + 1]
        today = today[~today.is_start]
        out = {"player": name, "phase": phase,
               "team": ctx.team_of(name),
               "ipl_batting_before_today": stats.batting_stats(name, ctx.date, phase),
               "ipl_bowling_before_today": stats.bowling_stats(name, ctx.date, phase)}
        if name in set(today.batter) | set(today.non_striker):
            out["batting_this_match_all_overs"] = stats.batting_stats(name, d=today)
        if name in set(today.bowler):
            out["bowling_this_match_all_overs"] = stats.bowling_stats(name, d=today)
        # Drop empty sections so the model isn't distracted by zeros.
        if out["ipl_bowling_before_today"]["matches"] == 0:
            out.pop("ipl_bowling_before_today")
        if out["ipl_batting_before_today"]["innings"] == 0:
            out.pop("ipl_batting_before_today")
        return out

    def head_to_head(batter: str, bowler: str) -> dict:
        """How a batter has fared against a specific bowler in IPL matches before today."""
        _log("head_to_head", batter=batter, bowler=bowler)
        b, w = ctx.resolve(batter), ctx.resolve(bowler)
        if not b or not w:
            return {"error": "Could not identify both players."}
        return stats.head_to_head(b, w, ctx.date)

    def recent_balls(n: int = 12) -> list:
        """Ball-by-ball commentary for the last n deliveries (max 36)."""
        _log("recent_balls", n=n)
        return ctx.replay.recent_balls(ctx.idx, min(max(int(n), 1), 36))

    def scorecard(innings: int) -> dict:
        """Batting and bowling scorecard for innings 1 or 2, up to the current ball."""
        _log("scorecard", innings=innings)
        bat, bowl = ctx.replay.scorecard(ctx.idx, int(innings))
        return {"batting": bat.to_dict("records"), "bowling": bowl.to_dict("records")}

    def venue_info() -> dict:
        """Venue history before today: average first-innings score, chasing win %, and
        the model's par score for this match."""
        _log("venue_info")
        v = stats.venue_stats(ctx.replay.meta.venue, ctx.date)
        v["model_par_score_today"] = round(float(ctx.replay.meta.par))
        v["head_to_head"] = stats.team_head_to_head(ctx.replay.team1, ctx.replay.team2, ctx.date)
        return v

    def what_if(extra_runs: int = 0, extra_wickets: int = 0, extra_balls: int = 6) -> dict:
        """Hypothetical: win probability if the batting side adds extra_runs and loses
        extra_wickets over the next extra_balls legal balls. E.g. 'what if they score 15
        off the next over without losing a wicket' -> extra_runs=15, extra_wickets=0,
        extra_balls=6."""
        _log("what_if", extra_runs=extra_runs, extra_wickets=extra_wickets, extra_balls=extra_balls)
        r = ctx.replay.row(ctx.idx).copy()
        hyp = pd.DataFrame([r])
        hyp["runs"] = r.runs + int(extra_runs)
        hyp["wkts"] = min(r.wkts + int(extra_wickets), 10)
        hyp["legal_balls"] = min(r.legal_balls + int(extra_balls), r.max_balls)
        hyp["runs_last_30"] = r.runs_last_30 + int(extra_runs)
        hyp["wkts_last_30"] = r.wkts_last_30 + int(extra_wickets)
        p_bat_now = r.p_team1 if r.batting_team == ctx.replay.team1 else 1 - r.p_team1
        p_bat_hyp = float(prob_batting_side(hyp)[0])
        return {"batting_team": r.batting_team, "win_prob_now": round(float(p_bat_now), 3),
                "win_prob_after": round(p_bat_hyp, 3),
                "scenario": f"+{extra_runs} runs, -{extra_wickets} wickets in {extra_balls} balls"}

    tools = [get_score, get_win_prob, player_stats, head_to_head, recent_balls, scorecard,
             venue_info, what_if]
    return [_json_safe(t) for t in tools]


def _json_safe(fn):
    """Return plain-JSON results (no numpy/pandas scalars) so the SDK can serialise them."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return wrapper


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def get_model_names() -> list[str]:
    configured = os.environ.get("GEMINI_MODEL")
    if not configured:
        try:
            import streamlit as st
            configured = st.secrets.get("GEMINI_MODEL")
        except Exception:
            configured = None
    return ([configured] if configured else []) + [m for m in DEFAULT_MODELS if m != configured]


def ask(question: str, ctx: MatchContext, history: list[dict] | None = None) -> dict:
    """Answer a fan question. Returns {"answer", "tools", "engine"} plus "error" when
    Gemini was configured but couldn't answer (the answer is then the offline one)."""
    ctx.tool_log.clear()
    key = get_api_key()
    if not key:
        return {"answer": offline_answer(question, ctx, reason="no_key"),
                "tools": list(ctx.tool_log), "engine": "offline"}
    try:
        answer, model = _ask_gemini(question, ctx, history or [], key)
        return {"answer": answer, "tools": list(ctx.tool_log), "engine": model}
    except GeminiError as e:  # overloaded, quota, bad key: degrade gracefully, never crash
        ctx.tool_log.clear()
        return {"answer": offline_answer(question, ctx, reason="gemini_failed"),
                "tools": list(ctx.tool_log), "engine": "offline", "error": str(e)}


# ---------------------------------------------------------------- Gemini calls
REQUEST_TIMEOUT_MS = 20_000  # per HTTP request; then fall through to the next model


class GeminiError(RuntimeError):
    """Gemini could not answer; the message is safe to show to the user."""


_working_model: str | None = None


@functools.lru_cache(maxsize=4)
def _client(key: str):
    from google import genai
    from google.genai import types
    # v1beta: on the default v1 API, gemini-2.5-flash rejects function calling with
    # "Function calling is not enabled for api version v1".
    # One attempt per model with a hard timeout: if a model is overloaded we'd rather move
    # on to the next model than sit through the SDK's own retries.
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        api_version="v1beta", timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=1)))


def _status(e: Exception) -> int | None:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


def _is_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (_status(e) in (401, 403) or "api key not valid" in msg or "api_key_invalid" in msg
            or "permission_denied" in msg or "unauthenticated" in msg)


def generate(key: str, contents, config=None, on_attempt=None) -> tuple[str, str]:
    """Call Gemini, trying models in order until one answers.

    Falls through to the next model on anything model-specific: not found / not enabled
    (404, 400), overloaded (503), rate-limited (429), server errors and timeouts. Stops
    straight away on an invalid key, since no model will work. The last model that
    worked is tried first next time.
    """
    global _working_model
    names = get_model_names()
    if _working_model in names:
        names = [_working_model] + [n for n in names if n != _working_model]
    client = _client(key)
    failures = []
    for name in names:
        if on_attempt:
            on_attempt(name)
        try:
            resp = client.models.generate_content(model=name, contents=contents, config=config)
            text = (resp.text or "").strip()
            if not text:
                failures.append(f"{name}: empty reply")
                continue
            _working_model = name
            return text, name
        except Exception as e:
            if _is_auth_error(e):
                raise GeminiError("Gemini rejected the API key. Check GEMINI_API_KEY in "
                                  ".streamlit/secrets.toml (or the app's Secrets on "
                                  "Streamlit Cloud).") from e
            code = _status(e)
            label = {404: "not available", 429: "rate limit reached", 503: "overloaded"}.get(
                code, f"{type(e).__name__} {code or ''}".strip())
            failures.append(f"{name}: {label}")
    raise GeminiError("All Gemini models failed (" + "; ".join(failures) + "). "
                      "Usually temporary; try again in a minute.")


def _ask_gemini(question, ctx, history, key) -> tuple[str, str]:
    from google.genai import types

    contents = []
    for turn in history[-8:]:
        if turn.get("engine") == "offline":
            continue  # template fallbacks aren't the model's own words; don't feed them back
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=make_tools(ctx),
        temperature=0.3,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8),
    )
    # A model that fails halfway may already have called tools; only show the winner's.
    return generate(key, contents, config, on_attempt=lambda _: ctx.tool_log.clear())


# ---------------------------------------------------------------- offline fallback
def offline_answer(question: str, ctx: MatchContext, reason: str = "no_key") -> str:
    """Template answers for the most common questions, built from the same tools."""
    tools = {f.__name__: f for f in make_tools(ctx)}
    q = question.lower()
    snap = tools["get_score"]()
    bat = snap["batting_team"]
    wp = snap["win_probability"]
    lead = max(wp, key=wp.get)
    situation = f"{bat} are {snap['score']} after {snap['overs']} overs"
    if snap["innings"] == 2 and not snap["match_finished"]:
        situation += (f", needing {snap['runs_needed']} off {snap['balls_left']} balls "
                      f"(required rate {snap['required_run_rate']}, current {snap['run_rate']})")

    player = _find_player_in_question(question, ctx)
    if player and not any(w in q for w in ("win", "chance", "worry", "ahead")):
        phase = next((p for p in ("powerplay", "middle", "death") if p in q), "all")
        s = tools["player_stats"](player, phase)
        parts = [f"**{s['player']}**" + (f" ({phase} overs)" if phase != "all" else "") + " in the IPL before today:"]
        bat_s, bowl_s = s.get("ipl_batting_before_today"), s.get("ipl_bowling_before_today")
        is_bowler = bool(bowl_s) and (not bat_s or int(bowl_s["overs"].split(".")[0]) * 6 > bat_s["balls"])
        want_bowl = "bowl" in q or "economy" in q or "wicket" in q or (is_bowler and "bat" not in q)
        if bowl_s and want_bowl:
            parts.append(f"{bowl_s['wickets']} wickets in {bowl_s['matches']} matches, economy "
                         f"{bowl_s['economy']}, dot balls {bowl_s['dot_ball_pct']}%.")
        if bat_s and not want_bowl:
            parts.append(f"{bat_s['runs']} runs in {bat_s['innings']} innings, strike rate "
                         f"{bat_s['strike_rate']}, average {bat_s['average']}.")
        if "batting_this_match_all_overs" in s:
            b = s["batting_this_match_all_overs"]
            parts.append(f"Today: {b['runs']} off {b['balls']}.")
        if "bowling_this_match_all_overs" in s:
            b = s["bowling_this_match_all_overs"]
            parts.append(f"Today: {b['wickets']}/{b['runs_conceded']} in {b['overs']} overs.")
        return " ".join(parts)

    if snap["match_finished"]:
        return f"The match is over: {snap['result']}."
    if "venue" in q or "pitch" in q or "ground" in q or "par" in q:
        v = tools["venue_info"]()
        return (f"At {v['venue']}, first innings have averaged {v['avg_first_innings']} "
                f"(last 10: {v['recent_avg_first_innings_last_10']}) and chasing teams win "
                f"{v['chasing_win_pct']}% of the time. The model's par score today is {v['model_par_score_today']}.")
    w = tools["get_win_prob"]()
    return (f"{situation}. MatchMate's model gives {lead} a {wp[lead]:.0%} chance of winning. "
            + (f"Biggest swing so far: {w['biggest_swings'][0]}" if w["biggest_swings"] else "")
            + OFFLINE_NOTE.get(reason, ""))


OFFLINE_NOTE = {
    "no_key": "\n\n_Quick data answer: add a Gemini API key for full conversational answers._",
    "gemini_failed": "\n\n_Gemini is busy right now, so here's a quick data answer. Try again shortly._",
}


def _find_player_in_question(question: str, ctx: MatchContext) -> str | None:
    words = re.findall(r"[A-Za-z]+", question)
    players = ctx.players_in_match()
    lower_map = {}
    for p in players:
        for w in p.split():
            if len(w) > 2:
                lower_map.setdefault(w.lower(), p)
    for w in words:
        if w.lower() in lower_map:
            return lower_map[w.lower()]
    # Not in this match: try capitalised words against all IPL players.
    stop = {"should", "how", "what", "who", "will", "does", "is", "the", "at", "in", "i", "worry",
            "run", "rate", "death", "powerplay", "bowl", "bat", "can", "win", "this", "match"}
    for w in words:
        if w[0].isupper() and w.lower() not in stop and len(w) > 3:
            hit = stats.resolve_player(w)
            if hit and w.lower() in hit.lower():
                return hit
    return None

