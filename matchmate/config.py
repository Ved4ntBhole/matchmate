"""Paths, team codes and name normalisation shared across MatchMate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "ipl"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LEADERBOARD_PATH = ROOT / "data" / "leaderboard.json"

DELIVERIES_PATH = PROCESSED_DIR / "deliveries.parquet"
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
TIMELINES_PATH = PROCESSED_DIR / "timelines.parquet"
OVER_RATES_PATH = PROCESSED_DIR / "over_base_rates.json"
MODEL_PATH = MODELS_DIR / "win_prob.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Seasons up to and including TRAIN_LAST_SEASON train the model; later seasons are
# held out. Every match replayed from the held-out seasons is out-of-sample.
TRAIN_LAST_SEASON = 2024

# Old franchise names are folded into the current ones so stats carry across rebrands.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

TEAM_CODES = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
    "Deccan Chargers": "DCG",
    "Pune Warriors": "PWI",
    "Rising Pune Supergiant": "RPS",
    "Gujarat Lions": "GL",
    "Kochi Tuskers Kerala": "KTK",
}

TEAM_COLORS = {
    "CSK": "#F9CD05", "MI": "#004BA0", "RCB": "#D11D2A", "KKR": "#3A225D",
    "SRH": "#F26522", "DC": "#17479E", "PBKS": "#DD1F2D", "RR": "#EA1A85",
    "GT": "#1B2133", "LSG": "#2ABFCB", "DCG": "#5B6770", "PWI": "#2F9BD6",
    "RPS": "#6F3C8F", "GL": "#E04F16", "KTK": "#7E3F98",
}

# Venue strings in Cricsheet vary ("..., Mumbai", renamed grounds). Map to one name.
VENUE_ALIASES = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",
}


def norm_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def team_code(name: str) -> str:
    return TEAM_CODES.get(name, "".join(w[0] for w in name.split()).upper())


def norm_venue(name: str) -> str:
    base = name.split(",")[0].strip()
    return VENUE_ALIASES.get(base, base)
