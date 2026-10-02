import unittest

import pandas as pd

from main import (
    clean_dataframe,
    validate_dataframe,
    remove_units_from_column,
)

from utils import same_file


class TestDataClean(unittest.TestCase):

    def test_duplicate_removal(self):
        df = pd.DataFrame(
            {
                "name": ["A", "B", "B"],
                "age": [20, 21, 21],
            }
        )

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=True,
            drop_missing=False,
            fill_numeric=None,
            fill_categorical=None,
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["duplicate_rows_removed"], 1)
        self.assertEqual(len(stats["duplicate_changes"]), 1)

        change = stats["duplicate_changes"][0]

        self.assertEqual(change["column"], "*")
        self.assertEqual(change["original_value"], "Duplicate row")
        self.assertEqual(change["cleaned_value"], "Removed")


    def test_drop_missing(self):
        df = pd.DataFrame(
            {
                "name": ["A", "B", None],
                "age": [20, None, 22],
            }
        )

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=False,
            drop_missing=True,
            fill_numeric=None,
            fill_categorical=None,
        )

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(stats["missing_rows_dropped"], 2)
        self.assertEqual(len(stats["missing_changes"]), 2)

        for change in stats["missing_changes"]:
            self.assertEqual(change["column"], "*")
            self.assertEqual(change["cleaned_value"], "Removed")
            self.assertEqual(
                change["reason"],
                "Dropped row containing missing value",
            )


    def test_fill_numeric_missing_audit(self):
        df = pd.DataFrame(
            {
                "age": [20, 30, None, 40],
                "marks": [70, None, 90, 80],
            }
        )

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=False,
            drop_missing=False,
            fill_numeric="median",
            fill_categorical=None,
        )

        self.assertFalse(cleaned["age"].isna().any())
        self.assertFalse(cleaned["marks"].isna().any())

        self.assertEqual(len(stats["missing_changes"]), 2)

        changes = stats["missing_changes"]

        self.assertEqual(changes[0]["column"], "age")
        self.assertEqual(changes[0]["original_value"], "NaN")
        self.assertEqual(changes[0]["cleaned_value"], 30.0)
        self.assertEqual(
            changes[0]["reason"],
            "Filled missing value using median",
        )

        self.assertEqual(changes[1]["column"], "marks")
        self.assertEqual(changes[1]["original_value"], "NaN")
        self.assertEqual(changes[1]["cleaned_value"], 80.0)


    def test_fill_categorical_missing_audit(self):
        df = pd.DataFrame(
            {
                "city": ["Bengaluru", "Mysuru", None, "Bengaluru"],
            }
        )

        cleaned, stats = clean_dataframe(
            df,
            remove_duplicates=False,
            drop_missing=False,
            fill_numeric=None,
            fill_categorical="mode",
        )

        self.assertFalse(cleaned["city"].isna().any())
        self.assertEqual(len(stats["missing_changes"]), 1)

        change = stats["missing_changes"][0]

        self.assertEqual(change["column"], "city")
        self.assertEqual(change["original_value"], "NaN")
        self.assertEqual(change["cleaned_value"], "Bengaluru")
        self.assertEqual(
            change["reason"],
            "Filled missing value using mode",
        )


    def test_same_file(self):
        self.assertTrue(
            same_file(
                "data/sample.csv",
                "data/./sample.csv",
            )
        )

        self.assertFalse(
            same_file(
                "data/sample.csv",
                "data/other.csv",
            )
        )


    def test_validation_min(self):
        df = pd.DataFrame(
            {
                "age": [10, 20, 30],
            }
        )

        rules = [
            ("min", "age", 15),
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["indexes"], [0])


    def test_validation_not_null(self):
        df = pd.DataFrame(
            {
                "email": ["a@test.com", None, "c@test.com"],
            }
        )

        rules = [
            ("not-null", "email", None),
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [1])


    def test_validation_max(self):
        df = pd.DataFrame(
            {
                "marks": [50, 80, 120],
            }
        )

        rules = [
            ("max", "marks", 100),
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [2])


    def test_validation_allowed_values(self):
        df = pd.DataFrame(
            {
                "city": ["Bengaluru", "Mysuru", "Delhi"],
            }
        )

        rules = [
            (
                "allowed-values",
                "city",
                ["Bengaluru", "Mysuru"],
            ),
        ]

        results = validate_dataframe(df, rules)

        self.assertEqual(results[0]["indexes"], [2])


    def test_remove_units_from_column(self):
        df = pd.DataFrame(
            {
                "weight": [
                    "70kg",
                    "60 kg",
                    "55kg",
                    "68",
                    None,
                ]
            }
        )

        changes = remove_units_from_column(
            df,
            "weight",
            "kg",
        )

        self.assertEqual(df.loc[0, "weight"], 70)
        self.assertEqual(df.loc[1, "weight"], 60)
        self.assertEqual(df.loc[2, "weight"], 55)

        self.assertEqual(df.loc[3, "weight"], "68")
        self.assertTrue(pd.isna(df.loc[4, "weight"]))

        self.assertEqual(len(changes), 3)

        self.assertEqual(
            changes[0]["reason"],
            "Removed kg unit",
        )


if __name__ == "__main__":
    unittest.main()