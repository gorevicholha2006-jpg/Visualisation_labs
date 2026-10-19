import requests
from bs4 import BeautifulSoup


def parse_web_page(url):
    """Return webpage text without HTML tags; raise HTTPError on failed status."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup.select("script, style, noscript, template"):
        element.decompose()
    return soup.get_text(separator=" ", strip=True)


if __name__ == "__main__":
    print(parse_web_page("https://books.toscrape.com/")[:255])
    print(parse_web_page("https://quotes.toscrape.com/")[:255])
