"""Days 2-3: train and evaluate the win-probability model.

Trains logistic regression and XGBoost separately for each innings on seasons up to
TRAIN_LAST_SEASON, evaluates on later seasons (out-of-sample), keeps the better model
per innings, then precomputes ball-by-ball win-probability timelines for every match.

Usage:  python scripts/train_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from matchmate.config import (DELIVERIES_PATH, MATCHES_PATH, METRICS_PATH, MODEL_PATH,  # noqa: E402
                              MODELS_DIR, OVER_RATES_PATH, TIMELINES_PATH, TRAIN_LAST_SEASON)
from matchmate.features import (INN1_FEATURES, INN1_MONOTONE, INN2_FEATURES,  # noqa: E402
                                INN2_MONOTONE, build_states, make_features)
from matchmate.winprob import prob_team1  # noqa: E402


def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mp, my = float(p[mask].mean()), float(y[mask].mean())
        rows.append({"bin": b, "mean_pred": mp, "frac_won": my, "n": int(mask.sum())})
        ece += mask.mean() * abs(mp - my)
    return rows, float(ece)


def scores(y, p):
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auc": float(roc_auc_score(y, p)),
    }


def make_models(inn: int):
    feats = INN1_FEATURES if inn == 1 else INN2_FEATURES
    mono = INN1_MONOTONE if inn == 1 else INN2_MONOTONE
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
        monotone_constraints=tuple(mono[f] for f in feats),
        eval_metric="logloss", n_jobs=4, random_state=7,
    )
    return {"logistic_regression": lr, "xgboost": xgb}


def phase(legal_balls):
    return np.select([legal_balls <= 36, legal_balls <= 90], ["powerplay", "middle"], "death")


def main() -> None:
    matches = pd.read_parquet(MATCHES_PATH)
    deliveries = pd.read_parquet(DELIVERIES_PATH)
    states = build_states(matches, deliveries)
    m = matches.set_index("match_id")
    states["season"] = states["match_id"].map(m["season"])
    states["winner"] = states["match_id"].map(m["winner"])
    states["method"] = states["match_id"].map(m["method"])
    states = states[states["winner"].notna()].copy()
    states["y"] = (states["winner"] == states["batting_team"]).astype(int)

    # First-innings model: only full 20-over games (rain-cut first innings are noise).
    # Second-innings model: rain-adjusted targets are fine, the state already reflects them.
    keep = (((states["innings"] == 2) & states["target"].notna()) |
            ((states["innings"] == 1) & states["method"].isna() & (states["max_balls"] == 120)))
    states = states[keep]
    # Drop decided chases: they are overridden by rules, not predicted.
    decided = (states["innings"] == 2) & ((states["runs"] >= states["target"]) |
                                          (states["wkts"] >= 10) |
                                          (states["legal_balls"] >= states["max_balls"]))
    states = states[~decided]

    train = states[states["season"] <= TRAIN_LAST_SEASON]
    test = states[states["season"] > TRAIN_LAST_SEASON]
    print(f"train rows {len(train):,} ({train.match_id.nunique()} matches), "
          f"test rows {len(test):,} ({test.match_id.nunique()} matches)")

    metrics = {"train_seasons": f"{int(train.season.min())}-{TRAIN_LAST_SEASON}",
               "test_seasons": f"{TRAIN_LAST_SEASON + 1}-{int(test.season.max())}",
               "train_matches": int(train.match_id.nunique()),
               "test_matches": int(test.match_id.nunique()),
               "innings": {}}
    best = {}
    for inn in (1, 2):
        tr, te = train[train.innings == inn], test[test.innings == inn]
        Xtr, Xte = make_features(tr, inn), make_features(te, inn)
        ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
        res = {}
        fitted = {}
        for name, model in make_models(inn).items():
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            res[name] = scores(yte, p)
            fitted[name] = (model, p)
            print(f"  innings {inn} {name:20s} " +
                  " ".join(f"{k}={v:.4f}" for k, v in res[name].items()))
        # Baseline: always predict the training base rate for that innings.
        base = np.full(len(yte), ytr.mean())
        res["baseline_base_rate"] = scores(yte, base) | {"auc": 0.5}
        pick = min(("logistic_regression", "xgboost"), key=lambda n: res[n]["log_loss"])
        model, p = fitted[pick]
        best[f"inn{inn}"] = model
        cal, ece = calibration(yte, p)
        ph = phase(te["legal_balls"].to_numpy())
        by_phase = {str(k): scores(yte[ph == k], p[ph == k]) for k in
                    ("powerplay", "middle", "death") if (ph == k).sum() > 0 and
                    len(np.unique(yte[ph == k])) > 1}
        feats = INN1_FEATURES if inn == 1 else INN2_FEATURES
        if pick == "xgboost":
            imp = dict(zip(feats, model.feature_importances_.round(4).tolist()))
        else:
            imp = dict(zip(feats, np.abs(model[-1].coef_[0]).round(4).tolist()))
        metrics["innings"][str(inn)] = {"models": res, "selected": pick, "ece": ece,
                                        "calibration": cal, "by_phase": by_phase,
                                        "feature_importance": imp}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)

    # Overall held-out numbers for the README headline.
    p_all = np.concatenate([best[f"inn{i}"].predict_proba(make_features(test[test.innings == i], i))[:, 1]
                            for i in (1, 2)])
    y_all = np.concatenate([test[test.innings == i]["y"].to_numpy() for i in (1, 2)])
    metrics["overall_test"] = scores(y_all, p_all)
    cal, ece = calibration(y_all, p_all)
    metrics["overall_test"]["ece"] = ece
    metrics["overall_calibration"] = cal
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("overall test:", {k: round(v, 4) for k, v in metrics["overall_test"].items()})

    build_timelines(matches, deliveries, best)
    build_over_rates(deliveries)


def build_timelines(matches, deliveries, model) -> None:
    """Precompute P(team1 wins) after every ball of every match for instant replay."""
    states = build_states(matches, deliveries)
    states["p_team1"] = prob_team1(states, model)
    cols = ["match_id", "seq", "innings", "is_start", "over", "ball", "legal", "batter",
            "bowler", "non_striker", "runs_batter", "runs_extras", "runs_total",
            "extra_type", "is_wicket", "bowler_wicket", "wicket_kind", "player_out",
            "runs", "wkts", "legal_balls", "max_balls", "target", "par",
            "runs_last_30", "wkts_last_30", "batting_team", "bowling_team", "p_team1"]
    tl = states[cols].sort_values(["match_id", "seq"]).reset_index(drop=True)
    tl["idx"] = tl.groupby("match_id").cumcount()
    tl.to_parquet(TIMELINES_PATH, index=False)
    print(f"timelines: {len(tl):,} rows")


def build_over_rates(deliveries) -> None:
    """Historical base rates for the over-prediction game, by over and wickets down."""
    d = deliveries
    overs = (d.groupby(["match_id", "innings", "over"])
             .agg(runs=("runs_total", "sum"), wkt=("is_wicket", "max"),
                  six=("runs_batter", lambda s: (s == 6).any()),
                  wkts_before=("inn_wkts", "first"), wk_first=("is_wicket", "first"),
                  legal=("legal", "sum"))
             .reset_index())
    overs = overs[overs["legal"] >= 6]
    overs["wkts_before"] = overs["wkts_before"] - overs["wk_first"].astype(int)
    overs["wb"] = pd.cut(overs["wkts_before"], [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str)
    rates = {}
    for (ov, wb), g in overs.groupby(["over", "wb"]):
        if len(g) < 30:
            continue
        rates[f"{ov}|{wb}"] = {"runs10": float((g.runs >= 10).mean()),
                               "wicket": float(g.wkt.mean()),
                               "six": float(g.six.mean()),
                               "n": int(len(g))}
    rates["_overall"] = {"runs10": float((overs.runs >= 10).mean()),
                         "wicket": float(overs.wkt.mean()), "six": float(overs.six.mean())}
    OVER_RATES_PATH.write_text(json.dumps(rates, indent=1))
    print(f"over base rates: {len(rates)} buckets")


if __name__ == "__main__":
    main()
