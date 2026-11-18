"""Day 1: parse raw Cricsheet JSON into processed parquet tables.

Usage:  python scripts/build_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matchmate.config import DELIVERIES_PATH, MATCHES_PATH, PROCESSED_DIR, RAW_DIR  # noqa: E402
from matchmate.data import build_tables  # noqa: E402
from matchmate.features import add_par_scores  # noqa: E402


def main() -> None:
    if not RAW_DIR.exists():
        sys.exit(f"Raw data not found at {RAW_DIR}. Run scripts/download_data.py first.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches, deliveries = build_tables()
    matches = add_par_scores(matches, deliveries)
    matches.to_parquet(MATCHES_PATH, index=False)
    deliveries.to_parquet(DELIVERIES_PATH, index=False)
    print(f"{len(matches)} matches, {len(deliveries)} deliveries "
          f"({matches.season.min()}-{matches.season.max()})")
    print(matches.groupby("season")["par"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
