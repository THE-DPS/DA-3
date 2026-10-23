from pathlib import Path
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

def main():
    target = RAW / "day.csv"
    if target.exists():
        print(f"Found existing file: {target}")
        return
    print(f"Downloading official UCI archive from {URL}")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        matches = [n for n in archive.namelist() if n.lower().endswith("/day.csv") or n.lower() == "day.csv"]
        if not matches:
            raise FileNotFoundError("day.csv was not found in the downloaded UCI archive.")
        target.write_bytes(archive.read(matches[0]))
    print(f"Saved {target} ({target.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
