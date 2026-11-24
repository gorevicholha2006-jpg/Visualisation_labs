"""Аналіз даних і моделі для прогнозу відтоку клієнтів."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parent / "resources" / "telco_churn.csv"
SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
LOGISTIC = "Логістична регресія"
TREE = "Дерево рішень"
FOREST = "Випадковий ліс"


def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


def data_audit(data):
    summary = data[NUMERIC].describe().T
    summary = summary.rename(columns={"50%": "median"})
    quartile_range = summary["75%"] - summary["25%"]
    lower = summary["25%"] - 1.5 * quartile_range
    upper = summary["75%"] + 1.5 * quartile_range
    summary["iqr_outliers"] = (
        data[NUMERIC].lt(lower) | data[NUMERIC].gt(upper)
    ).sum()
    overview = pd.Series({
        "rows": len(data),
        "columns": len(data.columns),
        "duplicate_rows": data.duplicated().sum(),
        "duplicate_customer_ids": data["customerID"].duplicated().sum(),
    })
    return overview, data.isna().sum().rename("missing"), summary


def prepare_data(data):
    data = data.drop_duplicates().copy()
    # Для нового клієнта без жодного місяця обслуговування сума ще не накопичилася.
    new_customers = data["TotalCharges"].isna() & data["tenure"].eq(0)
    data.loc[new_customers, "TotalCharges"] = 0
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    return data


def contract_summary(data):
    return data.groupby("Contract")["Churn"].agg(customers="size", churn_rate="mean")


def eda_plots(data):
    figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    counts = data["Churn"].value_counts().reindex([0, 1])
    bars = axis.bar(["Залишилися", "Пішли"], counts, color=["#38758a", "#b35139"])
    axis.bar_label(bars, padding=3)
    axis.set(title="Баланс класів", ylabel="Кількість клієнтів", ylim=(0, counts.max() * 1.16))
    yield "01_class_balance", figure

    figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    rates = contract_summary(data).loc[["Month-to-month", "One year", "Two year"], "churn_rate"] * 100
    bars = axis.bar(["Помісячний", "Один рік", "Два роки"], rates, color="#38758a")
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set(title="Відтік за типом контракту", ylabel="Клієнти, які пішли, %", ylim=(0, 55))
    yield "02_contract_churn", figure

    figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    for label, name, color in [(0, "Залишилися", "#38758a"), (1, "Пішли", "#b35139")]:
        axis.hist(
            data.loc[data["Churn"].eq(label), "MonthlyCharges"], bins=np.arange(15, 126, 5),
            density=True, alpha=0.50, label=name, color=color,
        )
    axis.set(title="Розподіл щомісячної плати", xlabel="Щомісячна плата", ylabel="Щільність")
    axis.legend()
    yield "03_monthly_charges", figure

    figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    axis.boxplot(
        [data.loc[data["Churn"].eq(label), "tenure"] for label in (0, 1)],
        tick_labels=["Залишилися", "Пішли"],
    )
    axis.set(title="Стаж клієнтів і відтік", ylabel="Місяці обслуговування")
    yield "04_tenure", figure

    figure, axis = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    corr = data[[*NUMERIC, "Churn"]].corr()
    image = axis.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = ["Стаж", "Плата / міс.", "Загальна плата", "Відтік"]
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Кореляція Пірсона")
    for (row, column), value in np.ndenumerate(corr.to_numpy()):
        axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                  color="white" if abs(value) > 0.65 else "black")
    figure.colorbar(image, ax=axis, shrink=0.8)
    yield "05_correlations", figure


def train_models(data):
    y = data["Churn"]
    x = data.drop(columns=["customerID", "Churn"])
    categorical = x.columns.difference(NUMERIC).tolist()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    models = {
        LOGISTIC: make_pipeline(
            clone(preprocessing), LogisticRegression(max_iter=1500, class_weight="balanced"),
        ),
        TREE: make_pipeline(clone(preprocessing), DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=SEED,
        )),
        FOREST: make_pipeline(preprocessing, RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED,
        )),
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    search = GridSearchCV(models[FOREST], {
        "randomforestclassifier__max_depth": [6, 12],
        "randomforestclassifier__min_samples_leaf": [2, 8],
    }, scoring="f1", cv=cv)
    search.fit(x_train, y_train)
    models[FOREST] = search.best_estimator_
    rows = []
    predictions = {}
    for name, model in models.items():
        if name == FOREST:
            cv_f1 = search.best_score_
        else:
            cv_f1 = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1").mean()
            model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[name] = predicted
        rows.append({
            "model": name,
            "cv_f1": cv_f1,
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
        })
    tuning = pd.DataFrame(search.cv_results_)[[
        "param_randomforestclassifier__max_depth",
        "param_randomforestclassifier__min_samples_leaf",
        "mean_test_score",
    ]]
    tuning.columns = ["max_depth", "min_samples_leaf", "mean_cv_f1"]
    return pd.DataFrame(rows).set_index("model"), tuning, y_test, predictions


def compare_algorithms(y_test, predictions):
    actual = y_test.to_numpy()
    logistic_correct = predictions[LOGISTIC] == actual
    forest_correct = predictions[FOREST] == actual
    logistic_only = int((logistic_correct & ~forest_correct).sum())
    forest_only = int((forest_correct & ~logistic_correct).sum())
    discordant = logistic_only + forest_only
    p_value = binomtest(logistic_only, discordant, p=0.5).pvalue if discordant else 1.0
    return pd.Series({
        "logistic_only_correct": logistic_only,
        "forest_only_correct": forest_only,
        "accuracy_difference_logistic_minus_forest": (logistic_only - forest_only) / len(actual),
        "p_value": p_value,
        "alpha": 0.05,
    })


def performance_plot(metrics):
    axis = metrics[["accuracy", "recall", "f1", "roc_auc"]].plot.bar(
        figsize=(8, 4.6), rot=0, color=["#38758a", "#b35139", "#58815d", "#8065a0"],
    )
    axis.set(title="Показники моделей на спільному тесті", xlabel="", ylim=(0, 1))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=4)
    axis.figure.tight_layout()
    return axis.figure


def confusion_plot(y_test, predictions, model_name):
    figure, axis = plt.subplots(figsize=(5.8, 4.5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[model_name], labels=[0, 1],
        display_labels=["Залишився", "Пішов"], cmap="Blues", colorbar=False, ax=axis,
    )
    axis.set(title=model_name, xlabel="Прогноз", ylabel="Фактичний клас")
    return figure
