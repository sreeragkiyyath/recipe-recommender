# What Can I Cook? — Recipe Recommender

A recipe recommender that takes the ingredients you already have and suggests recipes you can make — ranked by how few ingredients you're missing, not just raw similarity.

Runs with plain `python3` for the core logic and CLI — no third-party dependencies. A Flask API is available on top for programmatic access. Built to run standalone on your own server (e.g. an EC2 instance).

---

## Problem

Most recipe search tools assume you'll go buy what a recipe needs. This tool flips that: tell it what's already in your kitchen, and it finds recipes that fit — surfacing the ones where you're closest to having everything, and telling you exactly what's missing for the rest.

## Approach

1. **Data:** a self-contained sample of 30 recipes (`data/recipes_raw.csv`), ingredients normalized to strip quantities and units (`"3 cloves garlic, minced"` → `"garlic"`). Swap in a larger public dataset (Food.com / Epicurious) by matching the same `name,ingredients` CSV shape — no code changes needed.
2. **Matching:** TF-IDF vectorization of ingredient lists + cosine similarity between your input and every recipe — implemented from scratch in pure Python (no `scikit-learn`)
3. **Ranking:** re-ranked by missing-ingredient count first, similarity score second — a recipe you're 1 ingredient away from beats a "more similar" recipe you're 5 ingredients away from
4. **Model persistence:** the TF-IDF index is built once and saved to `data/model.json` (a "train once, serve many" split) rather than recomputed on every process start
5. **Interface:** a command-line tool, and a Flask API for programmatic/HTTP access

## Tech Stack

- Python 3.10+ standard library only for the core logic, data pipeline, model build, and CLI — no third-party dependencies there.
- Flask for the HTTP API layer (`app.py`) — the one place in this project with a dependency to install.

## Project Structure

```
.
├── README.md
├── CLAUDE.md                  # guidance for AI coding assistants working in this repo
├── requirements.txt            # flask — only needed for app.py
├── cli.py                      # CLI entry point
├── app.py                       # Flask API entry point
├── recommender.py             # matching + ranking logic (clean_ingredient, recommend_final)
├── data_prep.py                # cleaning + normalization pipeline
├── build_model.py              # builds + persists the TF-IDF index
├── test_recommender.py        # unittest suite
└── data/
    ├── recipes_raw.csv        # sample dataset (name + raw ingredient lines)
    ├── recipes_clean.json     # preprocessed dataset (generated, not hand-edited)
    └── model.json               # persisted TF-IDF index (generated, not hand-edited)
```

## Running Locally

**CLI** — no virtual environment or `pip install` needed, just a `python3` interpreter:

```bash
python3 data_prep.py            # builds recipes_clean.json from data/recipes_raw.csv
python3 build_model.py          # builds data/model.json from recipes_clean.json
python3 cli.py chicken rice onion garlic   # one-shot lookup
python3 cli.py                              # interactive mode
python3 cli.py --rebuild-model              # rebuild data/model.json
```

**API** — needs Flask installed:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py                   # serves on http://localhost:5000
```

Endpoints:

| Method | Path             | Body                                        | Description                              |
|--------|------------------|----------------------------------------------|-------------------------------------------|
| GET    | `/health`        | —                                            | Status + number of recipes loaded         |
| GET    | `/recipes`       | —                                            | List all recipes                          |
| POST   | `/recommend`     | `{"ingredients": [...], "top_n": 5}`        | Ranked recipe recommendations             |
| POST   | `/rebuild-model` | —                                            | Rebuilds the index from `recipes_clean.json` without restarting the server |

Example:
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice", "onion", "garlic"], "top_n": 5}'
```

## Running Tests

```bash
python3 -m unittest -v
```

## Evaluation / What I Tested

- Manually verified rankings against realistic ingredient lists before building the UI, e.g. `chicken, rice, onion, garlic` correctly surfaces `Garlic Rice` and `Garlic Mushroom Rice` (1–2 missing ingredients) ahead of recipes with higher raw text similarity but more missing ingredients, like `Chicken and Rice Soup` (4 missing).
- Spot-checked `clean_ingredient()` against messy real-world phrasing: `"2 cups all-purpose flour"` → `"all-purpose flour"`, `"3 cloves garlic, minced"` → `"garlic"`, `"salt to taste"` → `"salt"`, `"extra virgin olive oil"` → `"olive oil"`.
- Not yet tested with real users — next step before calling this done.

## What I Learned / Would Improve Next

- No ingredient synonym handling — "scallion" and "green onion" are treated as different ingredients; needs a mapping table.
- No quantity-awareness — having "1 egg" counts the same as having "6 eggs" for a recipe that needs 6.
- No dietary filtering (vegetarian, allergens, etc.).
- Sample dataset is only 30 recipes; swapping in a full Food.com/Epicurious dataset would give more realistic ranking behavior at scale.

## License

MIT
