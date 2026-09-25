import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from pathlib import Path

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "books_clean.csv"


def get_soup(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return BeautifulSoup(response.text, "html.parser")


def get_category_links():
    soup = get_soup(BASE_URL)
    categories = {}

    category_links = soup.select(
        "div.side_categories ul li ul li a"
    )

    for link in category_links:
        category_name = link.get_text(strip=True)
        category_url = urljoin(
            BASE_URL,
            link.get("href")
        )
        categories[category_name] = category_url

    return categories


def parse_price(price_text):
    if not price_text:
        return None

    price_text = price_text.strip()
    price_text = price_text.replace("£", "")
    price_text = price_text.replace(",", "")

    try:
        return float(price_text)
    except ValueError:
        return None


def parse_rating(rating_text):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    return rating_map.get(rating_text)


def parse_stock(availability_text):
    if not availability_text:
        return False

    return "in stock" in availability_text.lower()


def get_book_details(book_url, category):
    soup = get_soup(book_url)

    title_tag = soup.find("h1")
    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else None
    )

    price_tag = soup.select_one("p.price_color")
    price_text = (
        price_tag.get_text(strip=True)
        if price_tag
        else None
    )

    price_gbp = parse_price(price_text)

    availability_tag = soup.select_one(
        "p.availability"
    )

    availability = (
        availability_tag.get_text(
            " ",
            strip=True
        )
        if availability_tag
        else ""
    )

    in_stock = parse_stock(availability)

    rating_tag = soup.select_one(
        "p.star-rating"
    )

    rating_classes = (
        rating_tag.get("class", [])
        if rating_tag
        else []
    )

    star_rating = None

    for class_name in rating_classes:
        if class_name != "star-rating":
            star_rating = class_name
            break

    rating = parse_rating(star_rating)

    return {
        "title": title,
        "price_gbp": price_gbp,
        "star_rating": star_rating,
        "rating": rating,
        "availability": availability,
        "in_stock": in_stock,
        "category": category
    }


def scrape_category(category_name, category_url):
    books = []
    current_url = category_url

    while current_url:
        print(
            f"Scraping {category_name}: {current_url}"
        )

        soup = get_soup(current_url)

        book_links = soup.select(
            "article.product_pod h3 a"
        )

        for book_link in book_links:
            book_url = urljoin(
                current_url,
                book_link.get("href")
            )

            try:
                book = get_book_details(
                    book_url,
                    category_name
                )

                books.append(book)

            except requests.RequestException as error:
                print(
                    f"Error scraping {book_url}: {error}"
                )

        next_link = soup.select_one(
            "li.next a"
        )

        if next_link:
            current_url = urljoin(
                current_url,
                next_link.get("href")
            )
        else:
            current_url = None

    return books


def clean_data(df):

    def fix_encoding(value):
        if not isinstance(value, str):
            return value

        try:
            return value.encode(
                "latin1"
            ).decode("utf-8")
        except (
            UnicodeEncodeError,
            UnicodeDecodeError
        ):
            return value

    df["title"] = df["title"].apply(
        fix_encoding
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df["in_stock"] = df["in_stock"].astype(bool)

    df = df.dropna(
        subset=[
            "title",
            "category",
            "rating",
            "price_gbp"
        ]
    )

    df["rating"] = df["rating"].astype(int)

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    df = df[
        [
            "title",
            "price_gbp",
            "star_rating",
            "rating",
            "availability",
            "in_stock",
            "category",
            "price_inr"
        ]
    ]

    return df


def main():

    print("=" * 60)
    print("ZEpto DATA PIPELINE - BOOK SCRAPER")
    print("=" * 60)

    categories = get_category_links()

    print(
        f"Found {len(categories)} categories."
    )

    selected_categories = list(
        categories.items()
    )[:3]

    print("\nSelected categories:")

    for category_name, _ in selected_categories:
        print(f" - {category_name}")

    all_books = []

    for category_name, category_url in selected_categories:

        print(
            f"\nStarting category: {category_name}"
        )

        books = scrape_category(
            category_name,
            category_url
        )

        print(
            f"Books collected from {category_name}: "
            f"{len(books)}"
        )

        all_books.extend(books)

    df = pd.DataFrame(all_books)

    print(
        f"\nTotal scraped rows: {len(df)}"
    )

    df = df.drop_duplicates(
        subset=["title", "category"]
    )

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} books were scraped."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "At least 3 categories are required."
        )

    df = clean_data(df)

    if df["price_gbp"].isna().any():
        raise ValueError(
            "Some GBP prices are missing."
        )

    if df["price_inr"].isna().any():
        raise ValueError(
            "Some INR prices are missing."
        )

    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )
    except PermissionError:
        print(
            "\nPermission denied."
        )
        print(
            "Close books_clean.csv in Excel, "
            "VS Code, or any other program and "
            "run the script again."
        )
        return

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETED")
    print("=" * 60)

    print(f"\nTotal books: {len(df)}")

    print(
        f"Categories: {df['category'].nunique()}"
    )

    print("\nBooks per category:")

    print(
        df["category"].value_counts()
    )

    print("\nData types:")

    print(df.dtypes)

    print("\nSample data:")

    print(
        df.head(10).to_string(index=False)
    )

    print(
        f"\nCSV saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()