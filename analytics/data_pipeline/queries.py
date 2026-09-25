import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "output" / "books.db"
OUTPUT_FILE = BASE_DIR / "output" / "sql_results.txt"


def run_query(connection, name, query):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nSQL:")
    print(query)

    df = pd.read_sql_query(
        query,
        connection
    )

    print("\nOUTPUT:")
    print(df.to_string(index=False))

    return df


def main():

    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    q1 = """
    SELECT
        title,
        price_gbp,
        rating,
        in_stock
    FROM books
    WHERE price_gbp > 20
    """

    q2 = """
    SELECT
        title,
        price_gbp,
        rating
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10
    """

    q3 = """
    SELECT DISTINCT
        category_name
    FROM categories
    ORDER BY category_name
    """

    q4 = """
    SELECT
        title,
        price_gbp,
        price_inr
    FROM books
    WHERE price_gbp BETWEEN 10 AND 20
    ORDER BY price_gbp
    """

    q5 = """
    SELECT
        title,
        rating,
        in_stock
    FROM books
    WHERE rating IN (4, 5)
    ORDER BY rating DESC, title
    """

    q6 = """
    SELECT
        b.book_id,
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books AS b
    JOIN categories AS c
        ON b.category_id = c.category_id
    ORDER BY
        b.rating DESC,
        b.price_gbp DESC
    LIMIT 10
    """

    queries = [
        ("QUERY 1 - SELECT / WHERE", q1),
        ("QUERY 2 - ORDER BY / LIMIT", q2),
        ("QUERY 3 - DISTINCT", q3),
        ("QUERY 4 - BETWEEN", q4),
        ("QUERY 5 - IN", q5),
        ("QUERY 6 - JOIN", q6),
    ]

    output_parts = []

    for name, query in queries:

        df = run_query(
            connection,
            name,
            query
        )

        output_parts.append(
            "=" * 70
        )

        output_parts.append(name)

        output_parts.append(
            "=" * 70
        )

        output_parts.append(
            "SQL:\n" + query.strip()
        )

        output_parts.append(
            "\nOUTPUT:\n" +
            df.to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("PANDAS read_sql()")
    print("=" * 70)

    sql_df_1 = pd.read_sql(
        q2,
        connection
    )

    sql_df_2 = pd.read_sql(
        q6,
        connection
    )

    print("\nResult from Query 2:")
    print(sql_df_1.to_string(index=False))

    print("\nResult from JOIN query:")
    print(sql_df_2.to_string(index=False))

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merged_result = (
        merged_df[
            [
                "book_id",
                "title",
                "category_name",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
            ]
        ]
        .sort_values(
            ["rating", "price_gbp"],
            ascending=[False, False]
        )
        .head(10)
        .reset_index(drop=True)
    )

    sql_join_result = (
        sql_df_2
        .reset_index(drop=True)
    )

    sql_join_result["in_stock"] = (
        sql_join_result["in_stock"]
        .astype(bool)
    )

    merged_result["in_stock"] = (
        merged_result["in_stock"]
        .astype(bool)
    )

    print("\nSQL JOIN result:")
    print(sql_join_result.to_string(index=False))

    print("\nPandas merge result:")
    print(merged_result.to_string(index=False))

    equivalent = sql_join_result.equals(
        merged_result
    )

    print(
        f"\nAre SQL JOIN and pandas.merge equivalent? "
        f"{equivalent}"
    )

    output_parts.append(
        "\n" + "=" * 70
    )

    output_parts.append(
        "PANDAS read_sql / merge validation"
    )

    output_parts.append(
        "=" * 70
    )

    output_parts.append(
        "\nSQL JOIN result:\n" +
        sql_join_result.to_string(index=False)
    )

    output_parts.append(
        "\nPandas merge result:\n" +
        merged_result.to_string(index=False)
    )

    output_parts.append(
        f"\nEquivalent: {equivalent}"
    )

    OUTPUT_FILE.write_text(
        "\n\n".join(output_parts),
        encoding="utf-8"
    )

    print(
        f"\nAll SQL outputs saved to:\n{OUTPUT_FILE}"
    )

    connection.close()


if __name__ == "__main__":
    main()