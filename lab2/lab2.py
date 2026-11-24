import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker


DATA_PATH = Path(__file__).resolve().parent / "resources" / "data.json"


def extract_texts_from_file(file_path):
    with Path(file_path).open(encoding="utf-8") as file:
        data = json.load(file)
    texts = data.get("texts") if isinstance(data, dict) else None
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("JSON має містити поле texts зі списком рядків.")
    return texts


abbreviations = {
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "e.g.": "for example",
    "u.s.": "united states",
    "it's": "it is",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
}
abbreviation_pattern = re.compile(
    r"(?<![a-z])(?:" + "|".join(map(re.escape, abbreviations)) + r")(?![a-z])"
)
known_typos = {"netwroks": "networks", "swiming": "swimming"}
protected_words = {
    "oppenheimer", "christopher", "nolan", "cillian", "murphy",
    "manhattan", "dijkstra", "kruskal", "j",
}
spell = SpellChecker(language="en", distance=1)
stemmer = SnowballStemmer("english")


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = abbreviation_pattern.sub(lambda match: abbreviations[match.group()], text)
    text = re.sub(r"'s", "", text)
    return re.findall(r"[a-z]+", text)


def correct_spelling(word):
    if word in known_typos:
        return known_typos[word]
    if word in protected_words or len(word) < 4 or word in spell:
        return word
    candidates = spell.candidates(word) or set()
    return next(iter(candidates)) if len(candidates) == 1 else word


def process_texts(texts):
    return [
        " ".join(stemmer.stem(correct_spelling(word)) for word in tokenize(text))
        for text in texts
    ]


stemmed_stop_words = sorted({stemmer.stem(word) for word in ENGLISH_STOP_WORDS})


def fit_tfidf(texts_processed):
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        stop_words=stemmed_stop_words,
    )
    return vectorizer.fit_transform(texts_processed), vectorizer


def calculate_tfidf(texts_processed):
    if not texts_processed:
        return np.empty((0, 0))
    stop_words = set(stemmed_stop_words)
    if not any(set(text.split()) - stop_words for text in texts_processed):
        return np.zeros((len(texts_processed), len(texts_processed)))
    tfidf, _ = fit_tfidf(texts_processed)
    return np.clip(cosine_similarity(tfidf), 0, 1)


def visualize_data(cosine_matrix, texts_processed, title, *, signed=False):
    labels = [
        f"{number}. {' '.join(text.split()[:4])}"
        for number, text in enumerate(texts_processed, start=1)
    ]
    figure, axis = plt.subplots(figsize=(9, 7), layout="constrained")
    image = axis.imshow(
        cosine_matrix,
        vmin=-1 if signed else 0,
        vmax=1,
        cmap="coolwarm" if signed else "viridis",
    )
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8, label="Косинусна подібність")

    for (row, column), score in np.ndenumerate(cosine_matrix):
        red, green, blue, _ = image.cmap(image.norm(score))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axis.text(
            column, row, f"{score:.3f}", ha="center", va="center",
            color="black" if luminance > 0.55 else "white", fontsize=8,
        )
    return figure


def closest_pair(matrix):
    rows, columns = np.triu_indices(len(matrix), k=1)
    best = np.argmax(matrix[rows, columns])
    row, column = rows[best], columns[best]
    return row + 1, column + 1, matrix[row, column]


def get_texts():
    return [
        "NLP systems use AI to classify customer messages and detect sentiment.",
        "Natural language processing uses artificial intelligence to classify "
        "customer messages by sentiment.",
        "Ducks are swiming in ponds. Their webbed feet help them swim in shallow water.",
        "Ducks swim in shallow water and use webbed feet to move across ponds.",
        "Graph algorithms find the shortest path between nodes in transportation netwroks.",
        "In transportation networks, graph algorithms compute the shortest path between nodes.",
    ]


def train_word2vec(texts_processed):
    return Word2Vec(
        sentences=[text.split() for text in texts_processed],
        vector_size=50,
        window=4,
        min_count=1,
        sg=1,
        workers=1,
        seed=42,
        epochs=120,
    )


def embedding_similarity(texts_processed, model):
    weights, vectorizer = fit_tfidf(texts_processed)
    terms = vectorizer.get_feature_names_out()
    embeddings = np.stack([model.wv[word] for word in terms])
    weighted_sums = weights @ embeddings
    total_weights = np.asarray(weights.sum(axis=1))
    document_vectors = np.divide(
        weighted_sums,
        total_weights,
        out=np.zeros_like(weighted_sums),
        where=total_weights > 0,
    )
    return np.clip(cosine_similarity(document_vectors), -1, 1)
