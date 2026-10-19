import json
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_FILE = PROJECT_DIR / "result.json"
API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=48.2921&longitude=25.9358"
    "&start_date=2024-08-18&end_date=2024-08-20"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    "&timezone=Europe%2FKyiv"
)
CSV_API_URL = API_URL.replace("2024-08-18", "1997-05-22").replace(
    "2024-08-20", "1997-05-22"
)


def parse_api(api_url):
    """Save the API JSON response to result.json; raise HTTPError on failure."""
    response = requests.get(api_url, timeout=20)
    response.raise_for_status()
    data = response.json()
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parse_api(API_URL)
    print(f"Відповідь API збережено: {RESULT_FILE}")
