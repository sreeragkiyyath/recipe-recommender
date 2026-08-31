"""CLI entry point. No matching/ranking logic here — import from recommender.py.

Usage:
    python3 cli.py chicken rice onion garlic       # one-shot lookup
    python3 cli.py --top 3 chicken rice
    python3 cli.py                                  # interactive mode
    python3 cli.py --rebuild-model                  # rebuild data/model.json
"""

import argparse

from build_model import build_and_save
from recommender import build_index, load_index, load_recipes, recommend_final


def parse_args():
    parser = argparse.ArgumentParser(description="What Can I Cook? recipe recommender")
    parser.add_argument(
        "ingredients", nargs="*",
        help="Ingredients you have, e.g. chicken rice onion garlic. "
             "Omit to enter interactive mode.",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of recipes to show (default 5)")
    parser.add_argument(
        "--rebuild-model", action="store_true",
        help="Rebuild data/model.json from data/recipes_clean.json and exit.",
    )
    return parser.parse_args()


def print_results(results):
    if not results:
        print("No recipes found.")
        return
    for r in results:
        print(f"\n{r['name']}  (match {r['match_score']}%)")
        if r["missing"]:
            print(f"  Missing ({len(r['missing'])}): {', '.join(r['missing'])}")
        else:
            print("  You have everything you need!")


def main():
    args = parse_args()

    if args.rebuild_model:
        build_and_save()
        return

    recipes = load_recipes()
    try:
        idf, doc_vectors = load_index()
    except FileNotFoundError:
        print("No saved model found — building one in memory (run --rebuild-model to persist it).")
        idf, doc_vectors = build_index(recipes)

    if args.ingredients:
        results = recommend_final(
            args.ingredients, top_n=args.top, recipes=recipes, idf=idf, doc_vectors=doc_vectors
        )
        print_results(results)
        return

    print("What Can I Cook?")
    print("Enter ingredients you have, comma-separated (blank line to quit).\n")
    while True:
        raw = input("Ingredients: ").strip()
        if not raw:
            break
        user_ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        results = recommend_final(
            user_ingredients, top_n=args.top, recipes=recipes, idf=idf, doc_vectors=doc_vectors
        )
        print_results(results)
        print()


if __name__ == "__main__":
    main()
