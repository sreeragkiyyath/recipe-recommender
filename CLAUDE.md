# CLAUDE.md

Guidance for Claude Code (or any AI coding assistant) working in this repository.

## Project

"What Can I Cook?" — a recipe recommender. User enters ingredients they have; the app returns recipes ranked by how few additional ingredients they'd need to buy, not just raw text similarity.

This is a portfolio project. Priorities in order: (1) it actually works end-to-end, (2) the code is clean enough to walk an interviewer through, (3) it's genuinely usable by a non-technical person, not just a working prototype.

## Stack

- Python 3.10+, standard library only for the core logic (no pandas, no scikit-learn). Runs with a bare `python3` install, nothing to `pip install`.
- TF-IDF + cosine similarity reimplemented from scratch in `recommender.py` using `collections.Counter` and `math` — no heavier ML/embedding libraries needed for this project, keep it simple
- The TF-IDF index (`idf` + `doc_vectors`) is a persisted artifact (`data/model.json`), built via `build_model.py` — a train/build step separate from serving, rather than rebuilding it in-process every run
- Two interfaces: `cli.py` (stdlib only) and `app.py` (Flask — the **only** file in this project allowed to depend on a third-party package)
- No paid APIs, no LLMs, no external services requiring API keys — everything runs locally/free

## File Structure & Responsibilities

- `data_prep.py` — loads the raw recipe dataset (`csv` module), cleans ingredient text (strips quantities/units), outputs `data/recipes_clean.json`. Run this once, or whenever the raw dataset changes. Pure stdlib.
- `recommender.py` — core logic: `recommend_final(user_ingredients, top_n)`. Contains the from-scratch TF-IDF index (`build_index`), similarity scoring, missing-ingredient calculation, the re-ranking step (missing-count first, similarity second), and model persistence (`save_index`/`load_index`). Pure standard library, zero dependencies. This is the only module `cli.py` and `app.py` should import matching/ranking logic from.
- `build_model.py` — builds the TF-IDF index from `data/recipes_clean.json` and persists it to `data/model.json` via `build_and_save()`. Run once, or whenever the cleaned dataset changes. Pure stdlib. Both `cli.py --rebuild-model` and `app.py`'s `/rebuild-model` endpoint call this same function rather than duplicating the build logic.
- `cli.py` — UI only, stdlib only. Should not contain matching/ranking logic — import from `recommender.py`. Supports one-shot (`python3 cli.py chicken rice`), interactive, and `--rebuild-model` modes. Loads the persisted `data/model.json` if present, falling back to an in-memory `build_index()` (with a warning) if it isn't built yet.
- `app.py` — Flask API, UI only, exactly like `cli.py` but over HTTP. No matching/ranking logic — import from `recommender.py`. Holds recipes/idf/doc_vectors in a module-level dict guarded by a `threading.Lock` so `/rebuild-model` can safely swap them out while other requests are in flight.
- `test_recommender.py` — `unittest`-based tests against `recommender.py`, runnable with `python3 -m unittest`, no test framework dependency.
- `data/recipes_clean.json` — generated artifact, not hand-edited. Regenerate via `data_prep.py` if raw data changes.
- `data/model.json` — generated artifact, not hand-edited. Regenerate via `build_model.py` (or `cli.py --rebuild-model` / `POST /rebuild-model`) if `recipes_clean.json` changes.
- `requirements.txt` — only lists `flask`. Nothing else in the project needs an install.

## Conventions

- Ingredient text is always lowercased and stripped of quantities/units before comparison — use the `clean_ingredient()` helper in `recommender.py` for any new ingredient-handling code rather than re-implementing normalization inline.
- Keep `recommender.py` UI-agnostic and dependency-free — it should be testable/runnable directly, since debugging ranking logic in a plain script is much faster than through a UI, and both `cli.py` and `app.py` depend on it staying that way.
- Don't add a paid API, embedding model, or LLM call to this project. If a future feature genuinely needs one, flag it as a separate discussion rather than adding it directly — the goal here is a zero-cost, minimal-dependency project.
- Flask in `app.py` is the one deliberate exception to "no third-party dependencies," added explicitly for the API layer. Don't add further third-party dependencies (pandas, scikit-learn, Streamlit, extra Flask extensions, etc.) without checking with the user first — `recommender.py`, `data_prep.py`, `build_model.py`, and `cli.py` must all stay pure stdlib.

## Common Tasks

**Regenerate cleaned data after changing the raw dataset or cleaning logic:**
```bash
python3 data_prep.py
```

**Rebuild the persisted model after `recipes_clean.json` changes:**
```bash
python3 build_model.py          # or: python3 cli.py --rebuild-model
```

**Run the CLI locally (no install needed):**
```bash
python3 cli.py chicken rice onion garlic   # one-shot
python3 cli.py                              # interactive
```

**Run the Flask API locally:**
```bash
pip install -r requirements.txt   # only needed for this
python3 app.py                     # serves on :5000
```

**Run the test suite:**
```bash
python3 -m unittest -v
```

**Test the recommender logic directly (faster than going through the CLI or API):**
```python
from recommender import recommend_final
results = recommend_final(["chicken", "rice", "onion", "garlic"], top_n=5)
for r in results:
    print(r["name"], r["match_score"], r["missing"])
```

## Known Limitations (intentional, for now)

- No ingredient synonym handling (e.g. "scallion" vs "green onion" are treated as different) — noted as a future improvement in the README, not a bug to silently fix without updating the README.
- No quantity-awareness — having "1 egg" counts the same as having "6 eggs" for a recipe that needs 6.
- No dietary filtering (vegetarian, allergens, etc.).

If asked to fix any of these, treat it as a real feature addition — update the README's "What I'd improve next" section to reflect what was actually shipped.

## Deployment

Runs on the user's own EC2 instance rather than a managed platform.
- CLI: copy the repo over, run `python3 data_prep.py`/`python3 build_model.py` if the data or model need regenerating, and run `python3 cli.py`. No pip install step.
- API: `pip install -r requirements.txt` (Flask only), then `python3 app.py`. The dev server Flask ships with is not production-grade — for anything beyond local testing, front it with a production WSGI server (gunicorn, waitress, etc.) rather than running `app.run()` directly.
