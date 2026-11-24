"""Підготовка даних, навчання моделей і графіки для лабораторної №3."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

RESOURCES = Path(__file__).resolve().parent / "resources"
SEED = 42


@dataclass
class Experiment:
    model: object
    name: str
    metrics: dict
    x_test: pd.DataFrame | pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    split_sizes: tuple[int, int]


def read_data(filename):
    """Прочитати CSV або CSV.GZ з оригінальними заголовками."""
    return pd.read_csv(RESOURCES / filename)


def evaluate(x, y, model, name, *, classification=False, chronological=False):
    if chronological:
        cutoff = int(len(y) * 0.80)
        x_train, x_test = x.iloc[:cutoff], x.iloc[cutoff:]
        y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=SEED, stratify=y if classification else None,
        )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    if classification:
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
    else:
        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": root_mean_squared_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }

    return Experiment(
        model=model,
        name=name,
        metrics=metrics,
        x_test=x_test,
        y_test=y_test,
        predictions=predictions,
        split_sizes=(len(y_train), len(y_test)),
    )


def polynomial_model(degree):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression(),
    )


def co2_emissions():
    data = read_data("CO2_emission.csv").drop_duplicates().dropna(
        subset=["Engine_Size", "CO2_Emissions"],
    )
    return evaluate(
        data[["Engine_Size"]], data["CO2_Emissions"],
        polynomial_model(2), "Поліном 2-го степеня",
    )


def ice_cream_selling():
    feature, target = "Temperature (°C)", "Ice Cream Sales (units)"
    data = read_data("ice_cream_selling_data.csv").drop_duplicates().dropna(
        subset=[feature, target],
    )
    return evaluate(data[[feature]], data[target], polynomial_model(3), "Поліном 3-го степеня")


def power_consumption():
    features = ["Temperature", "Humidity", "WindSpeed"]
    targets = [f"PowerConsumption_Zone{zone}" for zone in (1, 2, 3)]
    data = read_data("powerconsumption.csv").dropna(subset=["Datetime", *features, *targets])
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="%m/%d/%Y %H:%M")
    data = data.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")
    model = HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, early_stopping=False, random_state=SEED,
    )
    return evaluate(
        data[features], data[targets].sum(axis=1), model,
        "Градієнтний бустинг", chronological=True,
    )


def heart_classification():
    data = read_data("heart.csv").dropna().drop_duplicates()
    numeric = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
    categorical = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
    prepare = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(prepare, LogisticRegression(max_iter=2000, random_state=SEED))
    return evaluate(
        data[numeric + categorical], data["output"], model,
        "Логістична регресія", classification=True,
    )


def clean_reviews(data):
    data = data[["Review", "Label"]].dropna().copy()
    data["Review"] = (
        data["Review"].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    data = data[data["Review"].ne("") & data["Label"].isin([1, 2, 3, 4, 5])]
    consistent = data.groupby("Review")["Label"].transform("nunique").eq(1)
    return data[consistent].drop_duplicates(subset=["Review"])


def classify_text_ml():
    reviews = clean_reviews(read_data("reviews.csv.gz"))
    model = make_pipeline(
        TfidfVectorizer(min_df=3, max_features=60_000, ngram_range=(1, 2)),
        LinearSVC(class_weight="balanced", random_state=SEED),
    )
    return evaluate(
        reviews["Review"], reviews["Label"], model, "Лінійний SVM", classification=True,
    )


def salary_prediction():
    data = read_data("Salary_dataset.csv")[["YearsExperience", "Salary"]].dropna().drop_duplicates()
    return evaluate(
        data[["YearsExperience"]], data["Salary"], LinearRegression(), "Лінійна регресія",
    )


def classification_details(result, labels, names):
    report = classification_report(
        result.y_test, result.predictions, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    report.pop("accuracy", None)
    return pd.DataFrame(report).T


def linear_parameters(result):
    """Повернути вільний член і ваги для вибраної лінійної або поліноміальної регресії."""
    model = result.model
    names = result.x_test.columns
    if isinstance(model, LinearRegression):
        regression = model
    elif hasattr(model, "named_steps") and "linearregression" in model.named_steps:
        regression = model.named_steps["linearregression"]
        names = model.named_steps["polynomialfeatures"].get_feature_names_out(names)
    else:
        return None
    return pd.Series([regression.intercept_, *regression.coef_], index=["intercept", *names])


def regression_plot(result, title, x_label, y_label):
    figure, axis = plt.subplots(figsize=(8.5, 4.7), layout="constrained")
    feature = result.x_test.columns[0]
    values = result.x_test[feature].to_numpy()
    grid = pd.DataFrame({feature: np.linspace(values.min(), values.max(), 250)})
    axis.scatter(values, result.y_test, alpha=0.45, label="Тестові спостереження")
    axis.plot(grid[feature], result.model.predict(grid), color="#a63825", label=result.name)
    axis.set(title=title, xlabel=x_label, ylabel=y_label)
    axis.legend()
    return figure


def power_plot(result):
    figure, (time_axis, scatter_axis) = plt.subplots(
        1, 2, figsize=(11, 4.6), layout="constrained",
    )
    # Тижневий фрагмент показує добові коливання. Праворуч наведено весь тест.
    count = min(len(result.y_test), 7 * 24 * 6)
    time_axis.plot(result.x_test.index[:count], result.y_test.iloc[:count], label="Факт")
    time_axis.plot(result.x_test.index[:count], result.predictions[:count], label="Прогноз")
    time_axis.set(title="Перший тиждень тесту", xlabel="Дата", ylabel="Сумарне споживання")
    time_axis.tick_params(axis="x", labelrotation=30)
    time_axis.legend()
    low = min(result.y_test.min(), result.predictions.min())
    high = max(result.y_test.max(), result.predictions.max())
    scatter_axis.scatter(result.y_test, result.predictions, s=6, alpha=0.15, edgecolors="none")
    scatter_axis.plot([low, high], [low, high], "--", color="#a63825", label="Ідеальний прогноз")
    scatter_axis.set(title="Уся тестова вибірка", xlabel="Фактичне", ylabel="Прогнозоване")
    scatter_axis.legend()
    figure.suptitle(f"Споживання електроенергії · {result.name}")
    return figure


def confusion_plot(result, title, labels, names):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for axis, normalize, subtitle, fmt in [
        (axes[0], None, "Кількість", "d"),
        (axes[1], "true", "Частка в межах справжнього класу", ".2f"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            result.y_test, result.predictions, labels=labels, display_labels=names,
            ax=axis, normalize=normalize, values_format=fmt, colorbar=False, cmap="Blues",
        )
        axis.set(title=subtitle, xlabel="Прогнозований клас", ylabel="Справжній клас")
    figure.suptitle(title)
    return figure
