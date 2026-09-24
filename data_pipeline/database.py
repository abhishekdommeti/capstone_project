import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "output" / "books_clean.csv"
DB_FILE = BASE_DIR / "output" / "books.db"


def create_database():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            "books_clean.csv not found. "
            "Run scraper.py first."
        )

    df = pd.read_csv(CSV_FILE)

    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    categories = sorted(
        df["category"].dropna().unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    category_map = {}

    rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    ).fetchall()

    for category_id, category_name in rows:
        category_map[category_name] = category_id

    for _, row in df.iterrows():

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                category_map[row["category"]],
            )
        )

    connection.commit()

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print("=" * 60)
    print("DATABASE CREATED")
    print("=" * 60)

    print(f"Books: {book_count}")
    print(f"Categories: {category_count}")
    print(f"Database: {DB_FILE}")

    connection.close()


if __name__ == "__main__":
    create_database()