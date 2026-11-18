# 2-minute demo script

Record on a phone-sized browser window (Chrome DevTools → device toolbar → iPhone 12), or
screen-record your real phone. Match: **Change match → Thrillers → PBKS vs KKR · 15 Apr 2025**.

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | The title, then the scoreboard | "Casual fans watching IPL often can't tell who's actually winning or why a wicket matters. Dream11 is now a second-screen platform, so I built MatchMate, an AI companion for the match on your TV." |
| 0:15–0:35 | Tap *Next over* a few times in innings 1, then show the chart moving | "This is a real 2025 match replayed ball by ball from Cricsheet's free data, as if it were live. That's a trick for prototyping without a paid live API. The model never saw any 2025 match." |
| 0:35–0:55 | *Jump → Innings break*, then *Next over* about 12 times until Russell's sixes; show the alert toast and Big moments | "When a ball swings the game, you get told in plain English: Russell's six lifted KKR's chances from 46% to 67%." |
| 0:55–1:20 | **Ask** tab: "Should I worry about this run rate?", then "How does Chahal do at the death?" Expand the data-used line | "Ask the match. Gemini answers through tool calls on real data: score, win probability, player stats. It never guesses, and it only knows what has happened so far, so no spoilers." |
| 1:20–1:40 | **Predict** tab: answer the 3 questions, lock in, tap *Next over*, show the points and leaderboard | "Between overs there's a free-to-play mini-game, just like Dream11's new format. The odds come from historical IPL overs, so bold calls pay more." |
| 1:40–1:55 | *Jump → End of match*, then **Recap** → *Recap the match* | "At full time you get a 5-line AI recap built from computed facts, including the turning point." |
| 1:55–2:00 | **Model** tab: calibration chart | "The model is calibrated on unseen seasons: 73% accuracy, 82% in chases. Built in a week with Claude Code, and it's live at this link." |

**Interview talking points:**
- **Replay as live:** the product is decoupled from the data source, so swapping in a paid
  feed is a one-module change.
- **Why logistic regression beat XGBoost:** the scoring era shifted, trees can't
  extrapolate, and era-relative features fixed it. Probabilities were measured by
  calibration, not just accuracy.
- **Grounded agent:** the tools read only data up to the current ball (there's a test
  proving it), and the app falls back offline if the LLM fails.
- **Product-first:** the README starts with the user journey, and every feature maps to
  the second-screen strategy.
