"""Win-probability inference: turn match states into P(team batting first wins)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .features import make_features


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def prob_batting_side(states: pd.DataFrame, model: dict | None = None) -> np.ndarray:
    """P(the side batting in that row wins), with hard overrides for decided chases."""
    model = model or load_model()
    out = np.full(len(states), np.nan)
    for inn in (1, 2):
        mask = (states["innings"] == inn).to_numpy()
        if not mask.any():
            continue
        X = make_features(states[mask], inn)
        out[mask] = model[f"inn{inn}"].predict_proba(X)[:, 1]

    # A chase that is already won or lost is certain, whatever the model says.
    inn2 = (states["innings"] == 2).to_numpy()
    need = (states["target"] - states["runs"]).to_numpy()
    balls_left = (states["max_balls"] - states["legal_balls"]).to_numpy()
    wkts = states["wkts"].to_numpy()
    won = inn2 & (need <= 0)
    over = inn2 & ~won & ((wkts >= 10) | (balls_left <= 0))
    out[over] = np.where(need[over] == 1, 0.5, 0.0)  # level scores -> super over
    out[won] = 1.0
    return np.clip(out, 0.001, 0.999)


def prob_team1(states: pd.DataFrame, model: dict | None = None) -> np.ndarray:
    p = prob_batting_side(states, model)
    return np.where(states["innings"].to_numpy() == 1, p, 1 - p)
