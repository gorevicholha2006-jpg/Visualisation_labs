import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lab2 import (
    DATA_PATH,
    calculate_tfidf,
    closest_pair,
    embedding_similarity,
    extract_texts_from_file,
    get_texts,
    process_texts,
    train_word2vec,
    visualize_data,
)

PROJECT_DIR = Path(__file__).resolve().parent


def save_results(matrix, texts, title, output_path, *, signed=False):
    document_ids = ",".join(f"document_{index}" for index in range(1, len(texts) + 1))
    np.savetxt(
        output_path.with_suffix(".csv"), matrix,
        delimiter=",", fmt="%.6f", header=document_ids, comments="",
    )
    figure = visualize_data(matrix, texts, title, signed=signed)
    figure.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(figure)

    first, second, score = closest_pair(matrix)
    print(f"{title} | {len(texts)} док. | найближча пара №{first} і №{second} ({score:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Лабораторна №2. Подібність текстів.")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "results",
        help="Каталог для CSV-матриць і PNG-графіків (типово results біля main.py).",
    )
    args = parser.parse_args()

    original = process_texts(extract_texts_from_file(DATA_PATH))
    own = process_texts(get_texts())
    model = train_word2vec(original + own)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, title, texts in [
        ("original", "Вихідні тексти", original),
        ("own", "Власні тексти", own),
    ]:
        save_results(
            calculate_tfidf(texts), texts, f"{title} · TF-IDF",
            args.output_dir / f"{name}_tfidf",
        )
        save_results(
            embedding_similarity(texts, model), texts, f"{title} · Word2Vec + TF-IDF",
            args.output_dir / f"{name}_word2vec", signed=True,
        )

    print(f"Результати збережено в {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
