"""Core matching/ranking logic for the recipe recommender.

Pure standard library — no third-party dependencies, no UI imports — so it
can be run and tested on its own with nothing but a plain `python3` install.
"""

import json
import math
import os
import re
from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes_clean.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")

# Units and descriptive words that carry no identity information about the
# ingredient itself and should be stripped during normalization.
_UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons",
    "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "gm", "gms", "kilogram", "kilograms", "kg", "kgs",
    "milligram", "milligrams", "mg", "milliliter", "milliliters", "ml", "liter",
    "liters", "l", "pinch", "pinches", "dash", "clove", "cloves", "slice",
    "slices", "can", "cans", "jar", "jars", "package", "packages", "pkg",
    "stick", "sticks", "head", "heads", "bunch", "bunches", "sprig", "sprigs",
    "quart", "quarts", "pint", "pints", "gallon", "gallons",
}

_DESCRIPTORS = {
    "fresh", "freshly", "chopped", "diced", "minced", "sliced", "grated",
    "shredded", "ground", "boneless", "skinless", "cooked", "uncooked", "raw",
    "ripe", "peeled", "crushed", "softened", "melted", "packed", "drained",
    "rinsed", "finely", "coarsely", "thinly", "whole", "large", "small",
    "medium", "extra", "virgin", "optional", "taste", "to", "of", "a", "an",
    "the", "and", "plus", "more", "for", "garnish",
}

_NUMERIC_TOKEN_RE = re.compile(r"^[\d¼½¾⅓⅔⅛⅜⅝⅞/.\-]+$")
_PAREN_RE = re.compile(r"\([^)]*\)")
_NON_ALNUM_EDGE_RE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")
_DIGIT_LETTER_RE = re.compile(r"(\d)([a-z])")  # split fused "100g" -> "100 g"


def clean_ingredient(text):
    """Normalize a raw ingredient string to a bare ingredient name.

    e.g. "2 cups all-purpose flour" -> "all-purpose flour"
         "3 cloves garlic, minced"  -> "garlic"
    """
    if not text:
        return ""

    text = text.lower()
    text = _PAREN_RE.sub(" ", text)
    text = text.split(",")[0]  # drop trailing descriptors after a comma
    text = _DIGIT_LETTER_RE.sub(r"\1 \2", text)  # "100g" -> "100 g", "8oz" -> "8 oz"

    tokens = text.split()
    kept = []
    for tok in tokens:
        tok = _NON_ALNUM_EDGE_RE.sub("", tok)
        if not tok:
            continue
        if _NUMERIC_TOKEN_RE.match(tok):
            continue
        if tok in _UNITS or tok in _DESCRIPTORS:
            continue
        kept.append(tok)

    return " ".join(kept).strip()


def load_recipes(path=DATA_PATH):
    """Load the cleaned recipe dataset produced by data_prep.py.

    Returns a list of {"name": str, "ingredients": list[str]} dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _term(ingredient):
    """Underscore a multi-word ingredient so it stays one term, e.g.
    "olive oil" -> "olive_oil"."""
    return ingredient.replace(" ", "_")


def _term_counts(ingredients):
    return Counter(_term(i) for i in ingredients if i)


def _normalize(vector):
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0:
        return vector
    return {term: weight / norm for term, weight in vector.items()}


def build_index(recipes):
    """Build a TF-IDF index over the recipe corpus.

    Returns (idf, doc_vectors): `idf` maps term -> inverse document
    frequency, `doc_vectors` is a list of unit-normalized TF-IDF vectors
    (dict term -> weight), one per recipe in `recipes` order. Cache and
    reuse both across calls to `recommend_final` to avoid rebuilding them
    on every lookup.
    """
    n_docs = len(recipes)
    doc_term_counts = [_term_counts(recipe["ingredients"]) for recipe in recipes]

    doc_frequency = Counter()
    for counts in doc_term_counts:
        doc_frequency.update(counts.keys())

    idf = {
        term: math.log((1 + n_docs) / (1 + freq)) + 1
        for term, freq in doc_frequency.items()
        
    }

    doc_vectors = [
        _normalize({term: count * idf[term] for term, count in counts.items()})
        for counts in doc_term_counts
    ]

    return idf, doc_vectors


def save_index(idf, doc_vectors, path=MODEL_PATH):
    """Persist a built index to disk so it doesn't need to be recomputed
    on every process start (e.g. by a long-running API server)."""
    payload = {"idf": idf, "doc_vectors": doc_vectors}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_index(path=MODEL_PATH):
    """Load a previously saved index. Raises FileNotFoundError if it hasn't
    been built yet — build it with `python3 build_model.py`."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["idf"], payload["doc_vectors"]


def _vectorize(ingredients, idf):
    counts = _term_counts(ingredients)
    vector = {term: count * idf[term] for term, count in counts.items() if term in idf}
    return _normalize(vector)


def _cosine_similarity(vector_a, vector_b):
    # Both vectors are already unit-normalized, so the dot product over
    # their shared terms *is* the cosine similarity.
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(weight * vector_b.get(term, 0.0) for term, weight in vector_a.items())


def recommend_final(user_ingredients, top_n=5, recipes=None, idf=None, doc_vectors=None):
    """Rank recipes for the given ingredients.

    Ranking is missing-ingredient count first, similarity score second —
    a recipe you're 1 ingredient away from beats a "more similar" recipe
    you're 5 ingredients away from.
    """
    if recipes is None:
        recipes = load_recipes()
    if idf is None or doc_vectors is None:
        idf, doc_vectors = build_index(recipes)

    user_clean = sorted({clean_ingredient(i) for i in user_ingredients} - {""})
    user_vector = _vectorize(user_clean, idf)
    user_set = set(user_clean)

    results = []
    for recipe, doc_vector in zip(recipes, doc_vectors):
        recipe_ingredients = set(recipe["ingredients"])
        missing = sorted(recipe_ingredients - user_set)
        have = sorted(recipe_ingredients & user_set)
        results.append({
            "name": recipe["name"],
            "match_score": round(_cosine_similarity(user_vector, doc_vector) * 100, 1),
            "have": have,
            "missing": missing,
            "missing_count": len(missing),
        })

    results.sort(key=lambda r: (r["missing_count"], -r["match_score"]))
    return results[:top_n]
