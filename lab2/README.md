# Лабораторна робота №2. Подібність текстів

Проєкт порівнює 10 текстів із завдання та 6 власних прикладів за допомогою TF‑IDF і косинусної подібності. Бонусна частина використовує Word2Vec зі зважуванням за TF‑IDF.

## Запуск із термінала

Потрібен **Python 3.12**. Відкрийте термінал у папці `Lab2_TFIDF`, де лежить `main.py`.

**macOS або Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Створення середовища й встановлення залежностей потрібні один раз. Для наступних запусків достатньо останньої команди. Після встановлення аналіз працює офлайн.

## Результати

За замовчуванням програма записує файли в папку `results/`. Кожна матриця має файл CSV і теплову карту PNG.

| Назви файлів | Що містять |
| --- | --- |
| `original_tfidf.*` | TF‑IDF для 10 текстів із завдання |
| `own_tfidf.*` | TF‑IDF для 6 власних текстів |
| `original_word2vec.*` | Word2Vec із вагами TF‑IDF для 10 текстів із завдання |
| `own_word2vec.*` | Word2Vec із вагами TF‑IDF для 6 власних текстів |

У CSV рядки й стовпці мають той самий порядок, що й документи. Скрипт також виводить найближчу пару за кожним методом. Для іншої папки результатів додайте `--output-dir шлях_до_папки` до команди запуску.

## Запуск у Jupyter

Відкрийте `assignement-2.ipynb` у Jupyter або VS Code з підтримкою Jupyter, виберіть ядро Python із середовища `.venv` і виконайте **Restart Kernel and Run All Cells**.

За потреби JupyterLab можна встановити в тому самому середовищі. На macOS або Linux після встановлення залежностей виконайте:

```bash
.venv/bin/python -m pip install jupyterlab
.venv/bin/python -m jupyterlab
```

На Windows PowerShell використовуйте `.\.venv\Scripts\python.exe` замість `.venv/bin/python`.

## Файли проєкту

- `main.py` запускає аналіз і зберігає результати.
- `lab2.py` містить підготовку текстів, TF‑IDF, Word2Vec і побудову теплових карт. Тексти для власного набору містяться у функції `get_texts()`.
- `assignement-2.ipynb` пояснює етапи роботи й містить збережені результати.
- `resources/data.json` містить вихідні тексти завдання.
- `requirements.txt` містить залежності Python.

[Завдання на GitHub](https://github.com/maxvonlancaster/ida-labs/blob/main/assignement-2.ipynb)
