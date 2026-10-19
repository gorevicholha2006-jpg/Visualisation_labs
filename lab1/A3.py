import json
from datetime import datetime
from pathlib import Path

from A2 import API_URL, CSV_API_URL, RESULT_FILE, parse_api


PROJECT_DIR = Path(__file__).resolve().parent
JSON_FILE = PROJECT_DIR / "weather.json"
JSON_DATE = "2024-8-19"
CSV_DATE = "1997-5-22"


def prepare_weather():
    """Download both periods and save the needed fields into weather.json."""
    records = []
    for api_url in (API_URL, CSV_API_URL):
        parse_api(api_url)
        with RESULT_FILE.open(encoding="utf-8") as file:
            data = json.load(file)
        hourly = data["hourly"]
        for i, time in enumerate(hourly["time"]):
            records.append({
                "date": time.split("T")[0],
                "time": time,
                "temperature_c": hourly["temperature_2m"][i],
                "humidity_percent": hourly["relative_humidity_2m"][i],
                "precipitation_mm": hourly["precipitation"][i],
                "wind_speed_kmh": hourly["wind_speed_10m"][i],
            })
    with JSON_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)


def parse_json(date):
    """Return a list of weather records for a date in YYYY-M-D format."""
    target_date = datetime.strptime(date, "%Y-%m-%d").date().isoformat()
    with JSON_FILE.open(encoding="utf-8") as file:
        records = json.load(file)
    return [record for record in records if record["date"] == target_date]


if __name__ == "__main__":
    if not JSON_FILE.exists():
        prepare_weather()
    print(json.dumps(parse_json(JSON_DATE), ensure_ascii=False, indent=4))
