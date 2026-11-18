"""Download Cricsheet's free IPL ball-by-ball data (JSON).

Usage:  python scripts/download_data.py
Cricsheet data is free under the Open Data Commons Attribution License.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matchmate.config import RAW_DIR  # noqa: E402

URL = "https://cricsheet.org/downloads/ipl_json.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.endswith(".json") and "/" not in n]
        for n in names:
            (RAW_DIR / n).write_bytes(z.read(n))
    print(f"Saved {len(names)} matches to {RAW_DIR}")


if __name__ == "__main__":
    main()
