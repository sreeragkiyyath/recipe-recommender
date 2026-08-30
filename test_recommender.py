"""Run with: python3 -m unittest"""

import unittest

from recommender import build_index, clean_ingredient, load_recipes, recommend_final


class CleanIngredientTests(unittest.TestCase):
    def test_strips_quantity_and_unit(self):
        self.assertEqual(clean_ingredient("2 cups all-purpose flour"), "all-purpose flour")

    def test_strips_descriptor_after_comma(self):
        self.assertEqual(clean_ingredient("3 cloves garlic, minced"), "garlic")

    def test_strips_to_taste(self):
        self.assertEqual(clean_ingredient("salt to taste"), "salt")

    def test_strips_fraction_quantity(self):
        self.assertEqual(clean_ingredient("1/2 tsp black pepper"), "black pepper")

    def test_strips_fused_metric_unit(self):
        self.assertEqual(clean_ingredient("100g potato"), "potato")
        self.assertEqual(clean_ingredient("10g sambar masala"), "sambar masala")

    def test_strips_fused_imperial_unit(self):
        self.assertEqual(clean_ingredient("8oz spaghetti"), "spaghetti")

    def test_strips_size_and_descriptor_words(self):
        self.assertEqual(clean_ingredient("extra virgin olive oil"), "olive oil")

    def test_empty_input(self):
        self.assertEqual(clean_ingredient(""), "")


class RecommendFinalTests(unittest.TestCase):
    def setUp(self):
        self.recipes = load_recipes()
        self.idf, self.doc_vectors = build_index(self.recipes)

    def _recommend(self, ingredients, top_n=5):
        return recommend_final(
            ingredients, top_n=top_n, recipes=self.recipes, idf=self.idf, doc_vectors=self.doc_vectors
        )

    def test_results_sorted_by_missing_count_first(self):
        results = self._recommend(["chicken", "rice", "onion", "garlic"])
        missing_counts = [r["missing_count"] for r in results]
        self.assertEqual(missing_counts, sorted(missing_counts))

    def test_exact_match_has_zero_missing(self):
        recipe = next(r for r in self.recipes if r["name"] == "Garlic Butter Chicken")
        results = self._recommend(recipe["ingredients"], top_n=1)
        self.assertEqual(results[0]["name"], "Garlic Butter Chicken")
        self.assertEqual(results[0]["missing"], [])

    def test_top_n_is_respected(self):
        results = self._recommend(["chicken", "rice"], top_n=3)
        self.assertEqual(len(results), 3)

    def test_unknown_ingredients_still_returns_results(self):
        results = self._recommend(["nonexistent-ingredient-xyz"])
        self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main()
