"""Flask API entry point. No matching/ranking logic here — import from recommender.py.

The only file in this project with a third-party dependency (Flask).
recommender.py, data_prep.py, build_model.py, and cli.py stay pure stdlib.

Run:
    pip install -r requirements.txt
    python3 build_model.py       # if data/model.json doesn't exist yet
    python3 app.py

Endpoints:
    GET  /health
    GET  /recipes
    POST /recommend       {"ingredients": [...], "top_n": 5}
    POST /rebuild-model
"""

import threading

from flask import Flask, jsonify, request

from build_model import build_and_save
from recommender import build_index, load_index, load_recipes, recommend_final

app = Flask(__name__)

_state_lock = threading.Lock()
_state = {"recipes": None, "idf": None, "doc_vectors": None}


@app.after_request
def _allow_cors(response):
    # Lets the standalone frontend (opened via file://) call this API.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _load_state():
    recipes = load_recipes()
    try:
        idf, doc_vectors = load_index()
    except FileNotFoundError:
        idf, doc_vectors = build_index(recipes)
    return recipes, idf, doc_vectors


def _init_state():
    recipes, idf, doc_vectors = _load_state()
    with _state_lock:
        _state["recipes"] = recipes
        _state["idf"] = idf
        _state["doc_vectors"] = doc_vectors


_init_state()


@app.get("/health")
def health():
    with _state_lock:
        loaded = _state["recipes"] is not None
        recipe_count = len(_state["recipes"]) if loaded else 0
    return jsonify({"status": "ok" if loaded else "not_ready", "recipes_loaded": recipe_count})


@app.get("/recipes")
def list_recipes():
    with _state_lock:
        recipes = _state["recipes"]
    return jsonify({"count": len(recipes), "recipes": recipes})


@app.post("/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    ingredients = body.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        return jsonify({"error": "'ingredients' must be a non-empty list of strings"}), 400

    top_n = body.get("top_n", 5)
    if not isinstance(top_n, int) or top_n < 1:
        return jsonify({"error": "'top_n' must be a positive integer"}), 400

    with _state_lock:
        recipes, idf, doc_vectors = _state["recipes"], _state["idf"], _state["doc_vectors"]

    results = recommend_final(ingredients, top_n=top_n, recipes=recipes, idf=idf, doc_vectors=doc_vectors)
    return jsonify({"results": results})


@app.post("/rebuild-model")
def rebuild_model():
    build_and_save()
    _init_state()
    with _state_lock:
        recipe_count = len(_state["recipes"])
    return jsonify({"status": "rebuilt", "recipes_loaded": recipe_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
