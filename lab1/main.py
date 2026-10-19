from A1 import parse_web_page
from A3 import CSV_DATE, JSON_DATE, parse_json, prepare_weather
from A4 import parse_csv, save_csv
from A5 import visualize_data
from A6 import run_lab


def main():
    print("A1: текст вебсторінок без HTML-тегів")
    for url in ("https://books.toscrape.com/", "https://quotes.toscrape.com/"):
        print(parse_web_page(url)[:255])

    print("A2–A3: API -> result.json -> weather.json")
    prepare_weather()
    print(f"JSON: за {JSON_DATE} знайдено {len(parse_json(JSON_DATE))} записів.")

    print("A4: вибір іншої дати -> CSV -> читання CSV")
    save_csv(parse_json(CSV_DATE))
    print(f"CSV: за {CSV_DATE} знайдено {len(parse_csv(CSV_DATE))} записів.")

    print("A5: графіки погоди")
    visualize_data()

    print("A6: імпутація, масштабування й графіки Kaggle")
    run_lab()


if __name__ == "__main__":
    main()
