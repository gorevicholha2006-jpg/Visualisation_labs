import csv
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "weather.csv"
FIELDS = ["date", "time", "temperature_c", "humidity_percent",
          "precipitation_mm", "wind_speed_kmh"]


def save_csv(records):
    """Save selected weather records with column headings into weather.csv."""
    with CSV_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def parse_csv(date):
    """Return a list of weather records from weather.csv for the given date."""
    target_date = datetime.strptime(date, "%Y-%m-%d").date().isoformat()
    with CSV_FILE.open(encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if row["date"] == target_date]


if __name__ == "__main__":
    from A3 import CSV_DATE, JSON_FILE, parse_json, prepare_weather
    if not JSON_FILE.exists():
        prepare_weather()
    save_csv(parse_json(CSV_DATE))
    print(parse_csv(CSV_DATE))
