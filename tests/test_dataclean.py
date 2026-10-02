import unittest

import pandas as pd
from main import clean_dataframe, same_file, validate_dataframe

class TestDataClean(unittest.TestCase):

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            "name": ["Rahul", "Priya", "Rahul"],
            "age": [20, 21, 20]
        })

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=True,
            drop_missing=False,
            fill_numeric=None,
            fill_categorical=None,
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["duplicate_rows_removed"], 1)

    def test_drop_missing(self):
        df = pd.DataFrame({
            "name": ["Rahul", "Priya", "Arjun"],
            "age": [20, 21, None]
        })

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=False,
            drop_missing=True,
            fill_numeric=None,
            fill_categorical=None,
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["missing_rows_dropped"], 1)
    def test_validation_min_rule(self):
        df = pd.DataFrame({
            "age": [20, 25, -5, 30]
        })

        rules = [
            ("min", "age", 0)
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [2])

    def test_validation_not_null_rule(self):
        df = pd.DataFrame({
            "city": ["Bengaluru", None, "Mysuru"]
        })

        rules = [
            ("not-null", "city", None)
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [1])
    def test_same_file(self):
        self.assertTrue(
            same_file("data/sample.csv", "data/sample.csv")
        )
    def test_validation_max_rule(self):
        df = pd.DataFrame({
            "marks": [85, 92, 105, 76]
        })

        rules = [
            ("max", "marks", 100)
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [2])

    def test_validation_allowed_values_rule(self):
        df = pd.DataFrame({
            "city": ["Bengaluru", "Mysuru", "Delhi"]
        })

        rules = [
            ("allowed-values", "city", ["Bengaluru", "Mysuru"])
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [2])

if __name__ == "__main__":
    unittest.main()