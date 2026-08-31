"""Builds data/recipes_clean.json from the raw recipe dataset.

Pure standard library — csv/json only. Run this once, or whenever
data/recipes_raw.csv changes:
    python3 data_prep.py
"""

import csv
import json
import os

from recommender import clean_ingredient

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes_raw.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes_clean.json")


def load_raw(path=RAW_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_recipes(raw_rows):
    cleaned = []
    for row in raw_rows:
        raw_parts = row["ingredients"].split("|")
        ingredients = sorted({clean_ingredient(part) for part in raw_parts} - {""})
        cleaned.append({"name": row["name"], "ingredients": ingredients})
    return cleaned


def main():
    raw_rows = load_raw()
    cleaned = clean_recipes(raw_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(cleaned)} recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
