import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from lab3 import (
    classification_details,
    classify_text_ml,
    co2_emissions,
    confusion_plot,
    heart_classification,
    ice_cream_selling,
    linear_parameters,
    power_consumption,
    power_plot,
    regression_plot,
    salary_prediction,
)

PROJECT = Path(__file__).resolve().parent


def save_experiment(output, key, result, figure, labels=None, names=None):
    figure.savefig(output / f"{key}.png", dpi=150)
    plt.close(figure)
    if labels is not None:
        report = classification_details(result, labels, names)
        report.to_csv(output / f"{key}_report.csv", index_label="class")
    else:
        parameters = linear_parameters(result)
        if parameters is not None:
            parameters.to_csv(output / f"{key}_coefficients.csv", header=["value"])

    scores = " | ".join(f"{name}={value:.3f}" for name, value in result.metrics.items())
    print(f"{key} · {result.name} | тест n={len(result.y_test)} | {scores}", flush=True)
    return {"task": key, "model": result.name, **result.metrics}


def main():
    parser = argparse.ArgumentParser(description="Лабораторна №3. Регресія й класифікація.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for key, task, plot_labels in [
        ("co2", co2_emissions, ("Викиди CO₂", "Об’єм двигуна, л", "Викиди CO₂")),
        ("ice_cream", ice_cream_selling, ("Продаж морозива", "Температура, °C", "Продаж, од.")),
        ("salary_bonus", salary_prediction, ("Зарплата і стаж", "Стаж, років", "Зарплата")),
    ]:
        result = task()
        rows.append(save_experiment(output, key, result, regression_plot(result, *plot_labels)))
        if key == "salary_bonus":
            example = pd.DataFrame({"YearsExperience": [5.0]})
            prediction = result.model.predict(example)[0]
            print(f"Бонус · прогноз для 5 років стажу = {prediction:.2f}", flush=True)

    result = power_consumption()
    rows.append(save_experiment(output, "power", result, power_plot(result)))

    for key, task, title, labels, names in [
        ("heart", heart_classification, "Класифікація міток heart.csv", [0, 1], ["0", "1"]),
        ("reviews", classify_text_ml, "Класифікація оцінок Coursera", [1, 2, 3, 4, 5],
         ["1", "2", "3", "4", "5"]),
    ]:
        result = task()
        figure = confusion_plot(result, f"{title} · {result.name}", labels, names)
        rows.append(save_experiment(output, key, result, figure, labels, names))

    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    print(f"Результати збережено в {output.resolve()}")


if __name__ == "__main__":
    main()
