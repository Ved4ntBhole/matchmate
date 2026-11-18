"""Core checks: data integrity, no-lookahead replay, model sanity, game scoring."""
import numpy as np
import pandas as pd
import pytest

from matchmate import game
from matchmate.agent import MatchContext, make_tools, offline_answer
from matchmate.replay import Replay, load_matches, load_timelines
from matchmate.stats import resolve_player
from matchmate.winprob import prob_batting_side

THRILLER = "1473468"  # PBKS defend 111 vs KKR, IPL 2025


@pytest.fixture(scope="module")
def replay():
    return Replay(THRILLER)


def test_innings_totals_match_result(replay):
    m = replay.meta
    assert m.winner == "Punjab Kings"
    assert replay.innings_total(1).startswith("Punjab Kings 111/10")
    assert replay.tl.legal_balls.max() <= 120


def test_final_probability_matches_winner():
    tl, m = load_timelines(), load_matches().set_index("match_id")
    last = tl.groupby("match_id").tail(1).set_index("match_id")
    decided = m[m.winner.notna() & (m.result != "tie")]
    last = last.loc[last.index.intersection(decided.index)]
    team1_won = decided.loc[last.index, "winner"] == decided.loc[last.index, "team1"]
    agree = ((last.p_team1 > 0.5) == team1_won).mean()
    assert agree > 0.99, agree


def test_monotonic_in_runs_needed():
    base = dict(innings=2, wkts=3, legal_balls=60, max_balls=120, par=180, target=180,
                runs_last_30=40, wkts_last_30=1)
    states = pd.DataFrame([{**base, "runs": r} for r in (60, 80, 100, 120)])
    p = prob_batting_side(states)
    assert np.all(np.diff(p) > 0), p  # more runs scored -> chaser more likely to win


def test_no_lookahead(replay):
    """Tools at ball idx must not change if everything after idx is different."""
    idx = 100
    ctx = MatchContext(replay, idx)
    before = make_tools(ctx)[0]()
    original = replay.tl
    replay.tl = original.copy()
    replay.tl.loc[idx + 1:, "runs_total"] = 6
    try:
        assert make_tools(MatchContext(replay, idx))[0]() == before
    finally:
        replay.tl = original


def test_swing_alert_text(replay):
    alerts = replay.swings(replay.last)
    assert alerts, "a thriller should have big moments"
    assert all("chances from" in a["text"] for a in alerts)


def test_game_settles_and_pays_odds(replay):
    picks = {"1-0": {"runs10": {"pick": False, "p_yes": 0.3}}}
    s = game.settle(picks, replay, replay.last)
    assert s["answered"] == 1
    p = picks["1-0"]["runs10"]
    assert p["points"] == (game.points_for(0.7) if p["correct"] else 0)
    assert game.points_for(0.5) == 10 and game.points_for(0.25) == 20


def test_upcoming_over_only_between_overs(replay):
    assert game.upcoming_over(replay, 0).over == 0
    assert game.upcoming_over(replay, 3) is None  # mid-over


def test_player_resolution():
    assert resolve_player("Bumrah") == "JJ Bumrah"
    assert resolve_player("Virat Kohli") == "V Kohli"
    assert resolve_player("MS Dhoni") == "MS Dhoni"


def test_offline_answer_is_grounded(replay):
    ans = offline_answer("Who is winning?", MatchContext(replay, 150))
    assert "%" in ans


# ---------------------------------------------------------------- Gemini fallback (no network)
class _FakeErr(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class _FakeModels:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        outcome = self.behaviour.get(model, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"text": f"answer from {model}"})()


def _fake_client(monkeypatch, behaviour):
    import matchmate.agent as A
    fake = type("C", (), {})()
    fake.models = _FakeModels(behaviour)
    monkeypatch.setattr(A, "_client", lambda key: fake)
    monkeypatch.setattr(A, "_working_model", None)
    monkeypatch.setattr(A, "get_model_names", lambda: ["m-busy", "m-missing", "m-good"])
    return A, fake.models


def test_generate_falls_through_overload_and_404(monkeypatch):
    A, models = _fake_client(monkeypatch, {"m-busy": _FakeErr(503, "overloaded"),
                                           "m-missing": _FakeErr(404, "not found")})
    text, used = A.generate("k", "hi")
    assert used == "m-good" and models.calls == ["m-busy", "m-missing", "m-good"]
    A.generate("k", "hi")  # the model that worked is tried first next time
    assert models.calls[-1] == "m-good" and len(models.calls) == 4


def test_generate_stops_on_bad_key(monkeypatch):
    A, models = _fake_client(monkeypatch, {"m-busy": _FakeErr(400, "API key not valid")})
    with pytest.raises(A.GeminiError, match="API key"):
        A.generate("k", "hi")
    assert models.calls == ["m-busy"]


def test_ask_degrades_to_offline_when_all_fail(monkeypatch, replay):
    A, _ = _fake_client(monkeypatch, {m: _FakeErr(503, "busy") for m in ["m-busy", "m-missing", "m-good"]})
    monkeypatch.setattr(A, "get_api_key", lambda: "k")
    res = A.ask("Who is winning?", A.MatchContext(replay, 150))
    assert res["engine"] == "offline" and "overloaded" in res["error"]
    assert "Gemini is busy" in res["answer"]
