import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent


def visualize_data():
    """Читає weather.csv, будує кілька графіків і зберігає їх у проєкті.

    Args:
        None.
    Returns:
        None.
    """
    with (PROJECT_DIR / "weather.csv").open(encoding="utf-8", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: row["time"])

    if not rows:
        raise ValueError("weather.csv не містить даних для графіків.")

    times = [datetime.fromisoformat(row["time"]) for row in rows]
    temperature = [float(row["temperature_c"]) for row in rows]
    humidity = [float(row["humidity_percent"]) for row in rows]
    precipitation = [float(row["precipitation_mm"]) for row in rows]
    wind = [float(row["wind_speed_kmh"]) for row in rows]
    period = times[0].strftime("%d.%m.%Y")
    if times[-1].date() != times[0].date():
        period += " – " + times[-1].strftime("%d.%m.%Y")
    output_dir = PROJECT_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    fig1, axes1 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(f"Чернівці · {period}\nТемпература й вологість")
    axes1[0].plot(times, temperature, color="#c45a18", marker="o", markersize=4)
    axes1[0].set_ylabel("Температура, °C")
    axes1[1].plot(times, humidity, color="#2475ad", marker="o", markersize=4)
    axes1[1].set_ylabel("Вологість, %")
    axes1[1].set_ylim(0, 100)

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig2.suptitle(f"Чернівці · {period}\nОпади й вітер")
    # Matplotlib вимірює ширину стовпчика дат у днях: 0.8 / 24 = 48 хвилин.
    axes2[0].bar(times, precipitation, width=0.8 / 24, color="#188c9e")
    axes2[0].set_ylabel("Опади за годину, мм")
    axes2[0].set_ylim(bottom=0)
    axes2[1].plot(times, wind, color="#8054a0", marker="o", markersize=4)
    axes2[1].set_ylabel("Швидкість вітру, км/год")
    axes2[1].set_ylim(bottom=0)

    for figure, axes, filename in [
        (fig1, axes1, "temperature_humidity.png"),
        (fig2, axes2, "precipitation_wind.png"),
    ]:
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d.%m"))
        axes[-1].set_xlabel("Місцевий час · Europe/Kyiv")
        figure.text(0.02, 0.01, "Джерело: Open-Meteo Historical Weather API (реаналіз)", fontsize=9)
        figure.tight_layout(rect=(0, 0.04, 1, 0.94))
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(f"Графіки збережено: {path}")


if __name__ == "__main__":
    visualize_data()
