"""MatchMate: an AI second-screen companion for live IPL cricket.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from matchmate import game, recap
from matchmate.agent import MatchContext, ask, get_api_key
from matchmate.config import METRICS_PATH, TEAM_COLORS, TRAIN_LAST_SEASON, team_code
from matchmate.replay import Replay, load_matches, load_timelines, overs_str

st.set_page_config(page_title="MatchMate", page_icon=":material/sports_cricket:",
                   layout="centered", initial_sidebar_state="collapsed",
                   menu_items={"About": "MatchMate: AI second-screen companion for IPL cricket."})


# ------------------------------------------------------------------ cached data
@st.cache_data
def matches_table() -> pd.DataFrame:
    m = load_matches()
    m = m[m.winner.notna() | (m.result == "tie")].copy()
    return m.sort_values("date", ascending=False)


@st.cache_resource(max_entries=32)
def get_replay(match_id: str) -> Replay:
    return Replay(match_id)


@st.cache_data
def thrillers(n: int = 10) -> list[str]:
    """Held-out matches where the eventual winner came back from the lowest odds."""
    tl, m = load_timelines(), load_matches().set_index("match_id")
    g = tl.groupby("match_id").p_team1.agg(["min", "max"]).join(m[["season", "team1", "winner"]])
    g = g[(g.season > TRAIN_LAST_SEASON) & g.winner.notna()]
    g["low"] = np.where(g.winner == g.team1, g["min"], 1 - g["max"])
    return g.nsmallest(n, "low").index.tolist()


@st.cache_data
def model_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def match_label(row) -> str:
    stage = row.stage if row.stage and not str(row.stage).startswith("Match") else ""
    return (f"{team_code(row.team1)} vs {team_code(row.team2)} · {row.date:%d %b %Y}"
            + (f" · {stage}" if stage else ""))


# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("idx", 0)
ss.setdefault("playing", False)
ss.setdefault("picks", {})      # match_id -> {over_key -> {question -> pick}}
ss.setdefault("chats", {})      # match_id -> [messages]
ss.setdefault("seen_alerts", 0)
ss.setdefault("fan_name", "")

matches = matches_table()

# ------------------------------------------------------------------ header + match picker
head = st.container(horizontal=True, vertical_alignment="center")
head.title(":material/sports_cricket: MatchMate", width="stretch")
picker = head.popover("Change match", icon=":material/swap_horiz:")
with picker:
    browse = st.segmented_control("Browse", ["Thrillers", "By season"], default="Thrillers",
                                  required=True, key="browse")
    if browse == "Thrillers":
        pool = matches[matches.match_id.isin(thrillers())]
        pool = pool.set_index("match_id").loc[thrillers()].reset_index()
        st.caption(f"The closest finishes and biggest comebacks of IPL "
                   f"{TRAIN_LAST_SEASON + 1}+, picked by the model. No spoilers.")
    else:
        seasons = sorted(matches.season.unique(), reverse=True)
        season = st.selectbox("Season", seasons, key="season")
        pool = matches[matches.season == season]
    labels = {r.match_id: match_label(r) for r in pool.itertuples()}
    match_id = st.selectbox("Match", list(labels), format_func=labels.get,
                            key=f"match_{browse}")
    speed = st.select_slider("Auto-play speed", options=[0.5, 1.0, 2.0, 3.0], value=1.0,
                             format_func=lambda s: f"{s:g}s per ball", key="speed")
    st.caption("Past matches are replayed ball by ball as if live. Seasons after "
               f"{TRAIN_LAST_SEASON} were never seen by the win-probability model.")
st.caption("Your AI second screen for live cricket")

if ss.get("loaded_match") != match_id:
    ss.loaded_match = match_id
    ss.idx = 0
    ss.playing = False
    ss.seen_alerts = 0

replay = get_replay(match_id)
ss.idx = min(ss.idx, replay.last)


def go(i: int) -> None:
    ss.idx = max(0, min(i, replay.last))
    if ss.idx >= replay.last:
        ss.playing = False


# ------------------------------------------------------------------ scoreboard


def scoreboard(idx: int) -> None:
    snap = replay.snapshot(idx)
    p1 = replay.p_team1(idx)
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center")
        top.markdown(f"**{replay.code1} vs {replay.code2}** · {replay.meta.venue}")
        if snap["match_finished"]:
            top.badge("Full time", color="gray")
        else:
            top.badge("Replaying live", icon=":material/radio_button_checked:", color="red")
        # One compact line so the whole scoreboard fits on a phone screen.
        st.markdown(f"### {team_code(snap['batting_team'])} {snap['score']} "
                    f"<small>({snap['overs']} ov)</small>", unsafe_allow_html=True)
        if snap["innings"] == 2:
            rrr = snap["required_run_rate"]
            st.markdown(f"Need **{snap['runs_needed']}** off **{snap['balls_left']}** · "
                        f"target {snap['target']} · run rate {snap['run_rate']:.2f}"
                        + (f" · required {rrr:.2f}" if rrr is not None else ""))
        else:
            st.markdown(f"Run rate **{snap['run_rate']:.2f}** · par score here "
                        f"**{snap['par_first_innings_score']}**",
                        help="Par: typical first-innings total at this venue in the current "
                             "scoring era, from matches before today.")
        st.markdown(f"**Win probability** · {replay.code1} **{p1:.0%}** – "
                    f"**{1 - p1:.0%}** {replay.code2}")
        st.progress(p1)
        chips = replay.current_over_chips(idx)
        line = snap["last_ball"]
        if chips:
            line = f"This over: `{' '.join(chips)}`  ·  {line}"
        st.caption(line)
        if snap["match_finished"]:
            st.success(snap["result"], icon=":material/emoji_events:")


scoreboard(ss.idx)

# ------------------------------------------------------------------ replay controls
ctl = st.container(horizontal=True)
if ctl.button("Restart", icon=":material/replay:", disabled=ss.idx == 0):
    go(0)
    ss.seen_alerts = 0
    st.rerun()
if ctl.button("Next ball", icon=":material/skip_next:", type="primary",
              disabled=replay.finished(ss.idx)):
    go(ss.idx + 1)
    st.rerun()
if ctl.button("Next over", icon=":material/fast_forward:", disabled=replay.finished(ss.idx)):
    go(replay.next_over_idx(ss.idx))
    st.rerun()
if ctl.button("Pause" if ss.playing else "Auto-play",
              icon=":material/pause:" if ss.playing else ":material/play_arrow:",
              disabled=replay.finished(ss.idx)):
    ss.playing = not ss.playing
    ss._last_tick = time.monotonic()
    st.rerun()
with ctl.popover("Jump", icon=":material/more_horiz:"):
    st.caption("Skipping ahead reveals the result.")
    if st.button("Innings break", disabled=replay.tl.innings.max() < 2):
        go(int(replay.tl.index[replay.tl.innings == 2][0]))
        st.rerun()
    if st.button("End of match"):
        go(replay.last)
        st.rerun()


@st.fragment(run_every=speed if ss.playing else None)
def autoplay_tick() -> None:
    """While auto-play is on, advance one ball per tick and redraw the page. The
    timestamp guard stops the fragment's run inside a full rerun from double-stepping."""
    if not ss.playing:
        return
    now = time.monotonic()
    if now - ss.get("_last_tick", 0.0) >= speed * 0.9:
        ss._last_tick = now
        go(ss.idx + 1)
        st.rerun(scope="app")


autoplay_tick()

# ------------------------------------------------------------------ swing alerts (toasts)
alerts = replay.swings(ss.idx)
for a in alerts[ss.seen_alerts:][-2:]:
    st.toast(a["text"], icon=":material/trending_up:" if a["delta"] > 0
             else ":material/trending_down:")
ss.seen_alerts = len(alerts)

# ------------------------------------------------------------------ views
view = st.segmented_control(
    "View", ["Live", "Ask", "Predict", "Recap", "Model"], default="Live", required=True,
    key="view", label_visibility="collapsed", width="stretch",
    format_func=lambda v: {"Live": ":material/monitoring: Live", "Ask": ":material/forum: Ask",
                           "Predict": ":material/target: Predict",
                           "Recap": ":material/summarize: Recap",
                           "Model": ":material/analytics: Model"}[v])


def win_prob_chart(idx: int) -> alt.Chart:
    tl = replay.tl.iloc[: idx + 1].copy()
    tl["x"] = tl["legal_balls"] / 6 + (tl["innings"] - 1) * 20
    tl["pct"] = tl["p_team1"] * 100
    tl["over"] = [f"Inn {i} · {overs_str(lb)}" for i, lb in zip(tl.innings, tl.legal_balls)]
    tl["event"] = [("Wicket" if w else ("Six" if rb == 6 else ("Four" if rb == 4 else "")))
                   for w, rb in zip(tl.is_wicket, tl.runs_batter)]
    c1 = TEAM_COLORS.get(replay.code1, "#3b6fd8")
    c2 = TEAM_COLORS.get(replay.code2, "#d8513b")
    base = alt.Chart(tl).encode(
        x=alt.X("x:Q", title="Overs (innings 1 → innings 2)",
                scale=alt.Scale(domain=[0, 40]),
                axis=alt.Axis(values=[0, 5, 10, 15, 20, 25, 30, 35, 40], labelExpr=
                              "datum.value <= 20 ? datum.value : datum.value - 20",
                              grid=False)))
    y = alt.Y("pct:Q", title=f"{replay.code1} win %", scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(values=[0, 25, 50, 75, 100], gridOpacity=0.3))
    above = base.mark_area(opacity=0.25, color=c1, interpolate="step-after").encode(
        y=alt.Y("pct:Q"), y2=alt.datum(50))
    line = base.mark_line(strokeWidth=2, color="#888", interpolate="step-after").encode(
        y=y,
        tooltip=[alt.Tooltip("over:N", title="Ball"),
                 alt.Tooltip("pct:Q", title=f"{replay.code1} win %", format=".0f"),
                 alt.Tooltip("runs:Q", title="Score"), alt.Tooltip("wkts:Q", title="Wkts"),
                 alt.Tooltip("event:N", title="Event")])
    wk = base.transform_filter("datum.is_wicket").mark_point(
        filled=True, size=60, color="#c0392b", stroke="white", strokeWidth=1).encode(
        y="pct:Q", tooltip=[alt.Tooltip("over:N", title="Wicket"),
                            alt.Tooltip("player_out:N", title="Out"),
                            alt.Tooltip("pct:Q", title=f"{replay.code1} win %", format=".0f")])
    mid = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="y:Q")
    brk = alt.Chart(pd.DataFrame({"x": [20]})).mark_rule(color="#bbb").encode(x="x:Q")
    labels = alt.Chart(pd.DataFrame({
        "x": [0.5, 0.5], "y": [95, 5], "t": [f"{replay.code1} ahead ↑", f"{replay.code2} ahead ↓"],
        "c": [c1, c2]})).mark_text(align="left", fontWeight="bold", fontSize=12).encode(
        x="x:Q", y="y:Q", text="t:N", color=alt.Color("c:N", scale=None))
    below = base.mark_area(opacity=0.25, color=c2, interpolate="step-after").encode(
        y=alt.Y("pct:Q"), y2=alt.datum(50))
    # Fill above 50 in team1 colour and below 50 in team2 colour.
    above = above.transform_calculate(pct="max(datum.p_team1 * 100, 50)")
    below = below.transform_calculate(pct="min(datum.p_team1 * 100, 50)")
    return (above + below + mid + brk + line + wk + labels).properties(height=260)


if view == "Live":
    st.altair_chart(win_prob_chart(ss.idx), alt=f"{replay.code1} win probability ball by ball")
    if alerts:
        st.subheader("Big moments", divider="gray")
        for a in reversed(alerts[-4:]):
            icon = ":material/trending_up:" if a["delta"] > 0 else ":material/trending_down:"
            st.markdown(f"{icon} **{a['over']}** (inn {a['innings']}) · {a['text']}")
    else:
        st.caption("Big win-probability swings will show up here as the match unfolds.")

    snap = replay.snapshot(ss.idx)
    if snap["batters"] or snap["bowler"]:
        b1, b2 = st.columns(2)
        with b1.container(border=True):
            st.markdown("**At the crease**")
            for b in snap["batters"]:
                st.markdown(f"{b['name']} · **{b['runs']}** ({b['balls']})")
        with b2.container(border=True):
            st.markdown("**Bowling**")
            if snap["bowler"]:
                bw = snap["bowler"]
                st.markdown(f"{bw['name']} · **{bw['wickets']}/{bw['runs']}** ({bw['overs']})")
    with st.expander("Commentary"):
        for line in reversed(replay.recent_balls(ss.idx, 18)):
            st.markdown(f"- {line}")
    with st.expander("Scorecard"):
        for inn in (1, 2):
            bat, bowl = replay.scorecard(ss.idx, inn)
            if bat.empty:
                continue
            st.markdown(f"**{replay.innings_total(inn) if inn == 1 or replay.finished(ss.idx) else snap['batting_team']}**")
            st.dataframe(bat, hide_index=True, alt=f"Innings {inn} batting")
            st.dataframe(bowl, hide_index=True, alt=f"Innings {inn} bowling")

elif view == "Ask":
    st.caption("Ask anything about the match. Answers come from match data via tool calls, "
               "and only know what has happened up to the current ball.")
    if not get_api_key():
        st.warning("No Gemini API key set, so answers use simple offline templates. See the "
                   "README to add one.", icon=":material/key_off:")
    msgs = ss.chats.setdefault(match_id, [])
    suggestions = {
        ":material/help: Should I worry about this run rate?": "Should I worry about this run rate?",
        ":material/sports_cricket: Who's winning and why?": "Who's winning right now, and why?",
        ":material/person: How does Bumrah do at the death?": "How does Bumrah do at the death?",
        ":material/stadium: Is this a good score here?": "Is this a good score at this venue?",
    }
    prompt = st.chat_input("Ask the match…", submit_mode="disable", key="ask_input")
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                st.caption("Data used: " + ", ".join(f"`{t}`" for t in msg["tools"]))
    if not msgs:
        holder = st.empty()
        pick = holder.pills("Try asking", list(suggestions), label_visibility="collapsed",
                            key=f"sugg_{match_id}")
        if pick:
            prompt = suggestions[pick]
        if prompt:
            holder.empty()
    if prompt:
        snap_over = overs_str(replay.row(ss.idx).legal_balls)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            ctx = MatchContext(replay, ss.idx)
            with st.status(":shimmer[Checking the match data]", type="compact") as status:
                res = ask(prompt, ctx, history=msgs)
                for t in res["tools"]:
                    st.write(f"`{t}`")
                status.update(label=f"Used {len(res['tools'])} data tool"
                                    f"{'s' if len(res['tools']) != 1 else ''} · {res['engine']}",
                              state="complete")
            st.markdown(res["answer"])
            if res.get("error"):
                st.warning(res["error"], icon=":material/cloud_off:")
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": res["answer"], "tools": res["tools"],
                     "at": snap_over, "engine": res["engine"]})

elif view == "Predict":
    picks = ss.picks.setdefault(match_id, {})
    summary = game.settle(picks, replay, ss.idx)
    open_picks = sum(1 for qs in picks.values() for p in qs.values() if not p.get("settled"))
    st.markdown(f"### :material/stars: {summary['points']} pts")
    st.caption(f"{summary['correct']}/{summary['answered']} correct · {open_picks} open picks · "
               "a correct pick pays 5 ÷ its probability, so bold calls pay more")

    ref = game.upcoming_over(replay, ss.idx)
    if ref is None:
        if replay.finished(ss.idx):
            st.info("Match over. Save your score below.", icon=":material/flag:")
        else:
            st.info("Predictions open between overs. Tap **Next over** to reach the end of "
                    "this over.", icon=":material/schedule:")
    elif ref.key in picks:
        st.success(f"Picks locked for {ref.label}. Play the over to see how you did.",
                   icon=":material/lock:")
    else:
        r = replay.row(ss.idx)
        nxt = replay.row(min(ss.idx + 1, replay.last))
        wkts = int(r.wkts) if int(nxt.innings) == int(r.innings) else 0
        batting = nxt.batting_team
        with st.container(border=True):
            st.markdown(f"**{ref.label}** · {team_code(batting)} batting")
            choices = {}
            for q, text in game.QUESTIONS.items():
                py = game.p_yes(q, ref.over, wkts)
                row = st.container(horizontal=True, vertical_alignment="center")
                choice = row.segmented_control(text, ["Yes", "No"],
                                               key=f"q_{match_id}_{ref.key}_{q}")
                st.caption(f"Yes in {py:.0%} of similar IPL overs · Yes pays "
                           f"{game.points_for(py)}, No pays {game.points_for(1 - py)}")
                if choice:
                    choices[q] = {"pick": choice == "Yes", "p_yes": py}
            if st.button("Lock in picks", type="primary", icon=":material/lock:",
                         disabled=not choices):
                picks[ref.key] = choices
                st.rerun()

    settled = [(k, q, p) for k, qs in picks.items() for q, p in qs.items() if p.get("settled")
               and not p.get("void")]
    if settled:
        st.subheader("Your picks", divider="gray")
        rows = [{"Over": f"Inn {k.split('-')[0]} · Ov {int(k.split('-')[1]) + 1}",
                 "Question": game.QUESTIONS[q], "Pick": "Yes" if p["pick"] else "No",
                 "Result": "Yes" if p["actual"] else "No",
                 "Points": p["points"]} for k, q, p in settled]
        st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, alt="Settled predictions")

    st.subheader("Match leaderboard", divider="gray")
    name = ss.fan_name or "You"
    board = [{"Fan": f"{name} (you)", "Points": summary["points"]}] + [
        {"Fan": f"{b} (bot)", "Points": p} for b, p in game.bot_scores(replay, ss.idx).items()]
    board = pd.DataFrame(board).sort_values("Points", ascending=False).reset_index(drop=True)
    board.index += 1
    st.dataframe(board, alt="Match leaderboard")
    st.caption("Bots are simulated fans who pick using historical rates; they play the same "
               "overs you are watching.")
    with st.expander("Hall of fame"):
        n = st.text_input("Display name", key="fan_name", max_chars=24)
        if st.button("Save my score", disabled=not n or summary["answered"] == 0):
            game.save_score(n, labels.get(match_id, match_id), summary["points"],
                            summary["correct"], summary["answered"])
            st.toast("Saved to the hall of fame", icon=":material/emoji_events:")
        hof = game.load_hall_of_fame()
        if hof:
            st.dataframe(pd.DataFrame(hof).rename(columns=str.title), hide_index=True,
                         alt="Hall of fame")
        else:
            st.caption("No saved scores yet.")

elif view == "Recap":
    finished = replay.finished(ss.idx)
    if not finished:
        st.info("The full recap unlocks at the end of the match. You can get a spoiler-free "
                "recap of what has happened so far.", icon=":material/lock_clock:")
    if st.button("Recap the match" if finished else "Recap so far", type="primary",
                 icon=":material/auto_awesome:"):
        with st.spinner("Writing the recap…"):
            ss[f"recap_{match_id}_{ss.idx}"] = recap.make_recap(replay, ss.idx)
    rc = ss.get(f"recap_{match_id}_{ss.idx}")
    if rc:
        with st.container(border=True):
            for line in rc["lines"]:
                st.markdown(line)
            st.caption(f"Written by {rc['engine']} from computed key facts")
        if rc.get("error"):
            st.warning(rc["error"], icon=":material/cloud_off:")
        with st.expander("Key facts used"):
            st.json(rc["facts"])

elif view == "Model":
    mt = model_metrics()
    o = mt["overall_test"]
    st.markdown(f"Trained on **IPL {mt['train_seasons']}** ({mt['train_matches']} matches), "
                f"tested on unseen **IPL {mt['test_seasons']}** ({mt['test_matches']} matches).")
    row = st.container(horizontal=True)  # wraps 2x2 on a phone instead of stacking 4 tall
    row.metric("Accuracy", f"{o['accuracy']:.1%}", border=True, width=140,
               help="Favourite (p ≥ 50%) went on to win")
    row.metric("AUC", f"{o['auc']:.3f}", border=True, width=140)
    row.metric("Brier", f"{o['brier']:.3f}", border=True, width=140,
               help="Mean squared error of probabilities; lower is better")
    row.metric("ECE", f"{o['ece']:.3f}", border=True, width=140,
               help="Expected calibration error; lower is better")
    cal = pd.DataFrame(mt["overall_calibration"])
    cal["Predicted"] = cal["mean_pred"] * 100
    cal["Actual"] = cal["frac_won"] * 100
    diag = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(
        strokeDash=[4, 4], color="#999").encode(x="x:Q", y="y:Q")
    pts = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=60, filled=True),
                                   strokeWidth=2).encode(
        x=alt.X("Predicted:Q", title="Predicted win % (binned)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Actual:Q", title="Actual win %", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("Predicted:Q", format=".0f"), alt.Tooltip("Actual:Q", format=".0f"),
                 alt.Tooltip("n:Q", title="Balls")])
    st.markdown("**Calibration on held-out seasons**: when the model says 70%, does that team "
                "win about 70% of the time?")
    st.altair_chart((diag + pts).properties(height=300), alt="Calibration curve")
    rows = []
    for inn, d in mt["innings"].items():
        for name, s in d["models"].items():
            rows.append({"Innings": inn, "Model": name.replace("_", " "),
                         "Log loss": round(s["log_loss"], 4), "Brier": round(s["brier"], 4),
                         "Accuracy": f"{s['accuracy']:.1%}", "AUC": round(s["auc"], 3),
                         "Selected": "✓" if name == d["selected"] else ""})
    st.markdown("**Model comparison** (held-out seasons, every ball)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Model comparison")
    st.caption("Logistic regression beat XGBoost out-of-sample: scoring jumped in the "
               "impact-player era (2023+), and the linear model on era-relative features "
               "extrapolates to those totals better than trees, which can't predict beyond "
               "the ranges they were trained on.")
