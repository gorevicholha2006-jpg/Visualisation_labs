import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lab4 import (
    compare_algorithms,
    confusion_plot,
    contract_summary,
    data_audit,
    eda_plots,
    load_data,
    performance_plot,
    prepare_data,
    train_models,
)


def save_plot(figure, path):
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Лабораторна №4. Відтік клієнтів Telco.")
    parser.add_argument("--output-dir", type=Path, default=project / "results")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    overview, missing, summary = data_audit(raw)
    overview.to_csv(output / "dataset_overview.csv", header=["value"])
    missing.to_csv(output / "missing_values.csv")
    summary.to_csv(output / "descriptive_statistics.csv")
    data = prepare_data(raw)
    contract_summary(data).to_csv(output / "contract_churn.csv")
    for name, figure in eda_plots(data):
        save_plot(figure, output / f"{name}.png")

    metrics, tuning, y_test, predictions = train_models(data)
    metrics.to_csv(output / "model_metrics.csv")
    tuning.to_csv(output / "forest_tuning.csv", index=False)
    selection = metrics["cv_f1"].idxmax()
    save_plot(performance_plot(metrics), output / "06_model_performance.png")
    save_plot(confusion_plot(y_test, predictions, selection), output / "07_confusion_matrix.png")

    test = compare_algorithms(y_test, predictions)
    test.to_csv(output / "mcnemar_test.csv", header=["value"])
    print(f"Клієнтів: {len(data)} | тестових записів: {len(y_test)}")
    print(metrics.round(3).to_string())
    print(f"Найвищий F1 на крос-валідації: {selection}")
    print(f"Двобічний тест Мак-Немара: p = {test['p_value']:.4f}")
    print(f"Результати збережено в {output.resolve()}")


if __name__ == "__main__":
    main()
