"""Kaggle -> imputation comparison -> normalization -> CSV files and plots."""
import json
from pathlib import Path
import textwrap

import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "imputation_results"
DATASET = "ilayaraja07/data-cleaning-feature-imputation/versions/1"
METHODS = ("mean", "median", "mode", "knn")
SCALERS = {"minmax": MinMaxScaler, "z-score": StandardScaler, "robust": RobustScaler}


def load_dataset(url, csv_file):
    """Load a Kaggle CSV as a DataFrame; reuse the saved local copy on later runs."""
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_file = raw_dir / csv_file
    if local_file.exists():
        return pd.read_csv(local_file)
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, url, csv_file)
    df.to_csv(local_file, index=False)
    return df


def numeric_columns(df):
    # Wine quality is an ordinal outcome, not a feature for imputation distances.
    return [c for c in df.select_dtypes(include="number") if c != "quality"]


def impute_data(df, method="knn"):
    """Fill numeric gaps with mean/median/mode/KNN; categorical gaps with mode."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    result = df.copy()
    columns = numeric_columns(df)
    if df[columns].isna().all().any():
        raise ValueError("Cannot estimate a completely missing numeric column.")
    if columns:
        if method == "knn":
            # Fit scaling on available values before calculating KNN distances.
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[columns])
            imputed = KNNImputer(n_neighbors=5, weights="distance").fit_transform(scaled)
            values = scaler.inverse_transform(imputed)
        else:
            strategy = "most_frequent" if method == "mode" else method
            values = SimpleImputer(strategy=strategy).fit_transform(df[columns])
        # Change only missing cells: observed values must remain exactly unchanged.
        for i, column in enumerate(columns):
            missing = df[column].isna()
            result.loc[missing, column] = values[missing, i]
    for column in df.select_dtypes(exclude="number"):
        modes = df[column].mode(dropna=True)
        replacement = modes.iloc[0] if len(modes) else "Unknown"
        result[column] = result[column].fillna(replacement)
    return result


def normalize_data(df, method="minmax"):
    """Scale numeric features; preserve categorical columns and Wine quality."""
    result = df.copy()
    columns = numeric_columns(df)
    result[columns] = SCALERS[method]().fit_transform(df[columns])
    return result


def compare_imputation(df):
    """Compare reconstruction on known values hidden with the same masks.

    Three fixed seeds; 10% of observed cells per numeric column; up to 1200 rows.
    NRMSE uses each column's observed standard deviation so units are comparable.
    This simulated MCAR experiment cannot measure error on real unknown values.
    """
    columns = numeric_columns(df)
    sample = df.sample(n=min(len(df), 1200), random_state=42)[columns].reset_index(drop=True)
    scale = sample.std(ddof=0).replace(0, 1).to_numpy()
    truth = sample.to_numpy(dtype=float)
    scores = {method: [] for method in METHODS}
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        mask = np.zeros(truth.shape, dtype=bool)
        for j in range(len(columns)):
            observed = np.flatnonzero(~np.isnan(truth[:, j]))
            chosen = rng.choice(observed, max(1, int(0.1 * len(observed))), replace=False)
            mask[chosen, j] = True
        corrupted = sample.mask(mask)
        for method in METHODS:
            prediction = impute_data(corrupted, method).to_numpy(dtype=float)
            errors = ((prediction - truth) / scale)[mask]
            scores[method].append(float(np.sqrt(np.mean(errors ** 2))))
    return pd.DataFrame([
        {"method": method, "nrmse_mean": np.mean(values),
         "nrmse_std": np.std(values, ddof=1)}
        for method, values in scores.items()
    ]).sort_values("nrmse_mean").reset_index(drop=True)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_data(df):
    """Save multiple Matplotlib visualizations of the normalized DataFrame."""
    folder = Path(df.attrs.get("output_dir", OUTPUT_DIR / "plots"))
    folder.mkdir(parents=True, exist_ok=True)
    columns = numeric_columns(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df[columns].to_numpy(), bins=20, label=columns, alpha=0.65)
    axes[0].set(title="Normalized feature distributions", xlabel="Scaled value", ylabel="Count")
    axes[0].legend(fontsize=7)
    corr = df[columns].corr()
    image = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [textwrap.fill(c, 12) for c in columns]
    axes[1].set_xticks(range(len(columns)), labels, rotation=60, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(columns)), labels, fontsize=7)
    axes[1].set_title("Feature correlation after processing")
    fig.colorbar(image, ax=axes[1], shrink=0.8)
    save_figure(fig, folder / "normalized_distribution_correlation.png")


def visualize_comparison(raw, filled, normalized, comparison, folder):
    columns = numeric_columns(raw)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(raw.columns))
    axes[0].bar(x - 0.2, raw.isna().sum(), width=0.4, label="Before")
    axes[0].bar(x + 0.2, filled.isna().sum(), width=0.4, label="After")
    axes[0].set_xticks(x, [textwrap.fill(c, 12) for c in raw.columns], rotation=60, ha="right", fontsize=7)
    axes[0].set(title="Real missing values", ylabel="Missing cells")
    axes[0].legend()
    axes[1].bar(comparison["method"], comparison["nrmse_mean"],
                yerr=comparison["nrmse_std"], capsize=5, color="#327ca5")
    axes[1].set(title="Artificial masking: 10%, 3 seeds", ylabel="NRMSE (lower is better)")
    save_figure(fig, folder / "missing_and_imputation_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column in zip(axes, columns[:3]):
        axis.hist(raw[column].dropna(), bins=20, alpha=0.5, label="Observed")
        axis.hist(filled[column], bins=20, histtype="step", linewidth=2, label="After imputation")
        axis.set(title=column, xlabel="Original units", ylabel="Count")
        axis.legend(fontsize=8)
    fig.suptitle("Observed values vs completed dataset")
    save_figure(fig, folder / "imputation_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    shown = columns[:4]
    for axis, (method, frame) in zip(axes, normalized.items()):
        axis.boxplot([frame[c] for c in shown], tick_labels=[textwrap.fill(c, 10) for c in shown],
                     flierprops={"markersize": 2, "alpha": 0.3})
        axis.set(title=method, ylabel="Scaled value")
        axis.tick_params(axis="x", labelrotation=40, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Normalization comparison (different vertical scales)")
    save_figure(fig, folder / "normalization_comparison.png")


def run_dataset(csv_file, label, normalization, reason):
    folder = OUTPUT_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(DATASET, csv_file)
    comparison = compare_imputation(raw)
    comparison.to_csv(folder / "imputation_comparison.csv", index=False)
    best = comparison.iloc[0]["method"]
    completed = {method: impute_data(raw, method) for method in METHODS}
    assert all(not frame.isna().any().any() for frame in completed.values())
    filled = completed[best]
    filled.to_csv(folder / "imputed.csv", index=False)
    normalized = {method: normalize_data(filled, method) for method in SCALERS}
    result = normalized[normalization]
    result.to_csv(folder / "normalized.csv", index=False)
    missing = pd.DataFrame({"before": raw.isna().sum(), "after": filled.isna().sum()})
    missing.to_csv(folder / "missing_values.csv")
    columns = numeric_columns(raw)
    q1, q3 = raw[columns].quantile(.25), raw[columns].quantile(.75)
    iqr = q3 - q1
    outliers = ((raw[columns] < q1 - 1.5 * iqr) | (raw[columns] > q3 + 1.5 * iqr)).sum()
    outliers.to_csv(folder / "iqr_outlier_counts.csv", header=["count"])
    assert not filled.isna().any().any()
    assert np.isfinite(result[columns].to_numpy()).all()
    for column in raw:
        known = raw[column].notna()
        assert raw.loc[known, column].equals(filled.loc[known, column])
    if "quality" in raw:
        assert result["quality"].equals(raw["quality"])
    result.attrs["output_dir"] = str(folder)
    visualize_data(result)
    visualize_comparison(raw, filled, normalized, comparison, folder)
    report = {
        "dataset": DATASET, "file": csv_file, "rows": len(raw), "columns": len(raw.columns),
        "real_missing_before": int(raw.isna().sum().sum()),
        "real_missing_after": int(filled.isna().sum().sum()),
        "numeric_missing_before": int(raw[columns].isna().sum().sum()),
        "selected_numeric_imputation": best, "categorical_imputation": "mode",
        "selected_normalization": normalization, "normalization_reason": reason,
        "comparison": comparison.to_dict(orient="records"),
        "evaluation_note": "Artificial MCAR masks on observed numeric values; not ground truth for real gaps.",
        "quality_note": "Wine quality is excluded from distances and scaling; no predictive model is trained.",
    }
    (folder / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=4))
    print(f"{label}: {len(raw)} rows; missing {report['real_missing_before']} -> 0; "
          f"numeric method={best}; scaling={normalization}")
    return report


def run_lab():
    """Run both datasets; keep final CSVs, comparison tables and plots."""
    datasets = [
        ("Students_Performance_knn.csv", "students", "minmax",
         "Оцінки мають спільну обмежену шкалу 0–100. MinMax переводить наявні діапазони в [0, 1]. "
         "Це min/max вибірки, а не ділення на теоретичні 100. Реальні пропуски лише у lunch: застосовано моду."),
        ("Wine_Quality.csv", "wine", "robust",
         "Числові показники мають різні одиниці, асиметрію й значення поза межами 1.5 IQR. "
         "RobustScaler використовує медіану та IQR, зменшуючи вплив крайніх значень на масштаб. "
         "Це обґрунтований вибір для завдання, а не доведений оптимум для прогнозування."),
    ]
    for csv_file, label, normalization, reason in datasets:
        run_dataset(csv_file, label, normalization, reason)
    print(f"Saved results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_lab()
