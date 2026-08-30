"""Builds data/model.json (the TF-IDF index) from data/recipes_clean.json.

This is the "training" step, separated from serving: build the index once,
persist it, and have the CLI/API load it instead of recomputing it on every
process start. Run this once, or whenever data/recipes_clean.json changes:
    python3 build_model.py
"""

from recommender import build_index, load_recipes, save_index


def build_and_save():
    recipes = load_recipes()
    idf, doc_vectors = build_index(recipes)
    save_index(idf, doc_vectors)
    print(f"Built index over {len(recipes)} recipes ({len(idf)} terms) -> data/model.json")


if __name__ == "__main__":
    build_and_save()
