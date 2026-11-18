"""Parse Cricsheet IPL JSON into tidy ball-by-ball and match tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, norm_team, norm_venue

# Dismissals that end a batter's innings and count as a wicket for the team.
NOT_TEAM_WICKETS = {"retired hurt", "retired not out"}
# Dismissals credited to the bowler.
NOT_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "retired not out",
                      "obstructing the field"}


def _season(info: dict) -> int:
    s = str(info.get("season"))
    # "2007/08" is the 2008 season, "2020/21" is the 2020 season held in the UAE.
    if s == "2007/08":
        return 2008
    if s == "2009/10":
        return 2010
    if s == "2020/21":
        return 2020
    return int(s[:4])


def parse_match(path: Path) -> tuple[dict, list[dict]] | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    info = d["info"]
    outcome = info.get("outcome", {})
    winner = outcome.get("winner") or outcome.get("eliminator")
    innings = [inn for inn in d.get("innings", []) if not inn.get("super_over")]
    if len(innings) < 1:
        return None

    match_id = path.stem
    team1 = norm_team(innings[0]["team"])
    teams = [norm_team(t) for t in info["teams"]]
    team2 = teams[1] if teams[0] == team1 else teams[0]

    target = innings[1].get("target", {}) if len(innings) > 1 else {}
    by = outcome.get("by", {})
    if "runs" in by:
        margin = f"by {by['runs']} runs"
    elif "wickets" in by:
        margin = f"by {by['wickets']} wickets"
    else:
        margin = ""
    if outcome.get("eliminator"):
        margin = "in a Super Over"
    if outcome.get("method"):
        margin += f" ({outcome['method']})"

    match = {
        "match_id": match_id,
        "season": _season(info),
        "date": info["dates"][0],
        "venue": norm_venue(info.get("venue", "Unknown")),
        "city": info.get("city", ""),
        "team1": team1,
        "team2": team2,
        "toss_winner": norm_team(info.get("toss", {}).get("winner", "")),
        "toss_decision": info.get("toss", {}).get("decision", ""),
        "winner": norm_team(winner) if winner else None,
        "result": outcome.get("result", "normal"),
        "margin": margin.strip(),
        "method": outcome.get("method"),
        "player_of_match": ", ".join(info.get("player_of_match", [])),
        "stage": str(info.get("event", {}).get("stage", "") or f"Match {info.get('event', {}).get('match_number', '')}"),
        "target_runs": target.get("runs"),
        "target_overs": target.get("overs"),
        "overs": info.get("overs", 20),
    }

    rows = []
    for inn_no, inn in enumerate(innings[:2], start=1):
        batting = norm_team(inn["team"])
        bowling = team2 if batting == team1 else team1
        runs = wkts = legal = 0
        for over in inn.get("overs", []):
            ball_in_over = 0
            for dl in over["deliveries"]:
                extras = dl.get("extras", {})
                is_legal = not ("wides" in extras or "noballs" in extras)
                if is_legal:
                    legal += 1
                    ball_in_over += 1
                r = dl["runs"]
                runs += r["total"]
                wk = dl.get("wickets", [])
                kinds = [w["kind"] for w in wk]
                team_wk = sum(k not in NOT_TEAM_WICKETS for k in kinds)
                wkts += team_wk
                rows.append({
                    "match_id": match_id,
                    "innings": inn_no,
                    "batting_team": batting,
                    "bowling_team": bowling,
                    "over": over["over"],
                    "ball": ball_in_over,
                    "legal": is_legal,
                    "batter": dl["batter"],
                    "bowler": dl["bowler"],
                    "non_striker": dl["non_striker"],
                    "runs_batter": r["batter"],
                    "runs_extras": r["extras"],
                    "runs_total": r["total"],
                    "extra_type": next(iter(extras), None),
                    "is_wicket": team_wk > 0,
                    "bowler_wicket": any(k not in NOT_BOWLER_WICKETS for k in kinds),
                    "wicket_kind": kinds[0] if kinds else None,
                    "player_out": wk[0]["player_out"] if wk else None,
                    "inn_runs": runs,
                    "inn_wkts": wkts,
                    "legal_balls": legal,
                })
    return match, rows


def build_tables(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, rows = [], []
    for path in sorted(raw_dir.glob("*.json")):
        parsed = parse_match(path)
        if parsed is None:
            continue
        m, r = parsed
        matches.append(m)
        rows.extend(r)
    mdf = pd.DataFrame(matches)
    mdf["date"] = pd.to_datetime(mdf["date"])
    mdf = mdf.sort_values(["date", "match_id"]).reset_index(drop=True)
    ddf = pd.DataFrame(rows)
    ddf["seq"] = ddf.groupby("match_id").cumcount()
    return mdf, ddf
