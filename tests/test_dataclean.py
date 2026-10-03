import os
import unittest
import tempfile
import pandas as pd

from cleaner import (
    clean_dataframe,
    remove_units_from_column,
    standardize_text_column,
)
from utils import same_file
from commands import run_clean
from validator import validate_dataframe
from operation_executor import execute_operation
from instruction_parser import parse_instruction
from instruction_normalizer import normalize_instruction
from operations import (
    get_operation,
    is_implemented_operation,
    is_supported_operation,
    list_operations,
    validate_operation,
    create_operation_request,
)


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

    def test_standardize_text_title(self):
        df = pd.DataFrame(
            {
                "gender": ["male", "FEMALE", " Male ", "Female"],
            }
        )

        changes = standardize_text_column(
            df,
            "gender",
            "title",
        )

        self.assertEqual(
            df["gender"].tolist(),
            ["Male", "Female", "Male", "Female"],
        )

        self.assertEqual(len(changes), 3)

    def test_standardize_text_strip(self):
        df = pd.DataFrame(
            {
                "city": [" Bengaluru ", "Mysuru", " Bengaluru"],
            }
        )

        changes = standardize_text_column(
            df,
            "city",
            "strip",
        )

        self.assertEqual(
            df["city"].tolist(),
            ["Bengaluru", "Mysuru", "Bengaluru"],
        )

        self.assertEqual(len(changes), 2)

    def test_run_clean_rejects_overwriting_input(self):
        with self.assertRaises(SystemExit):
            run_clean(
                "data/sample.csv",
                False,
                False,
                None,
                None,
                None,
                None,
                "data/sample.csv",
                False,
            )

    def test_run_clean_preview_does_not_create_output(self):
        output_path = "data/test_preview_output.csv"

        if os.path.exists(output_path):
            os.remove(output_path)

        try:
            run_clean(
                "data/sample.csv",
                False,
                False,
                "median",
                None,
                None,
                None,
                output_path,
                True,
            )

            self.assertFalse(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_supported_operation(self):
        self.assertTrue(is_supported_operation("standardize_text"))

    def test_unsupported_operation(self):
        self.assertFalse(is_supported_operation("unknown_operation"))

    def test_get_operation(self):
        operation = get_operation("remove_units")

        self.assertIsNotNone(operation)
        self.assertTrue(operation["requires_column"])
        self.assertIn("unit", operation["parameters"])

    def test_list_operations(self):
        operations = list_operations()

        self.assertIn("remove_duplicates", operations)
        self.assertIn("standardize_text", operations)
        self.assertIn("date_format", operations)
        self.assertIn("currency_format", operations)

    def test_implemented_operation(self):
        self.assertTrue(is_implemented_operation("standardize_text"))

        self.assertFalse(is_implemented_operation("date_format"))

    def test_valid_operation_parameters(self):
        valid, message = validate_operation(
            "standardize_text",
            {"style": "title"},
        )

        self.assertTrue(valid)
        self.assertEqual(
            message,
            "Operation is valid.",
        )

    def test_invalid_operation_parameters(self):
        valid, message = validate_operation(
            "standardize_text",
            {"style": "random"},
        )

        self.assertFalse(valid)
        self.assertIn(
            "Invalid value",
            message,
        )

    def test_missing_operation_parameter(self):
        valid, message = validate_operation(
            "standardize_text",
            {},
        )

        self.assertFalse(valid)
        self.assertIn(
            "Missing required parameter",
            message,
        )

    def test_unimplemented_operation(self):
        valid, message = validate_operation(
            "date_format",
            {"format": "MM-DD-YYYY"},
        )

        self.assertFalse(valid)
        self.assertIn(
            "not implemented yet",
            message,
        )

    def test_create_operation_request(self):
        request = create_operation_request(
            "standardize_text",
            column="gender",
            parameters={"style": "title"},
        )

        self.assertEqual(
            request,
            {
                "operation": "standardize_text",
                "column": "gender",
                "parameters": {"style": "title"},
            },
        )

    def test_create_operation_request_requires_column(self):
        with self.assertRaises(ValueError):
            create_operation_request(
                "standardize_text",
                parameters={"style": "title"},
            )

    def test_create_operation_request_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            create_operation_request(
                "standardize_text",
                column="gender",
                parameters={"style": "random"},
            )

    def test_parse_remove_duplicates_instruction(self):
        request = parse_instruction("Remove duplicates")

        self.assertEqual(
            request,
            {
                "operation": "remove_duplicates",
                "column": None,
                "parameters": {},
            },
        )

    def test_parse_remove_duplicate_instruction(self):
        request = parse_instruction("Remove duplicate")

        self.assertEqual(
            request,
            {
                "operation": "remove_duplicates",
                "column": None,
                "parameters": {},
            },
        )

    def test_parse_remove_units_instruction(self):
        request = parse_instruction("Remove kg from the weight column")

        self.assertEqual(
            request,
            {
                "operation": "remove_units",
                "column": "weight",
                "parameters": {
                    "unit": "kg",
                },
            },
        )

    def test_parse_unknown_instruction(self):
        with self.assertRaises(ValueError):
            parse_instruction("Do something random")

    def test_parse_fill_numeric_median_instruction(self):
        request = parse_instruction("Fill missing numeric values using median")

        self.assertEqual(
            request,
            {
                "operation": "fill_numeric",
                "column": None,
                "parameters": {
                    "strategy": "median",
                },
            },
        )

    def test_parse_fill_numeric_mean_instruction(self):
        request = parse_instruction("Fill missing numeric values using mean")

        self.assertEqual(
            request,
            {
                "operation": "fill_numeric",
                "column": None,
                "parameters": {
                    "strategy": "mean",
                },
            },
        )

    def test_parse_invalid_fill_numeric_strategy(self):
        with self.assertRaises(ValueError):
            parse_instruction("Fill missing numeric values using mode")

    def test_execute_remove_duplicates(self):
        df = pd.DataFrame(
            {
                "name": ["A", "B", "B"],
                "age": [20, 21, 21],
            }
        )

        request = {
            "operation": "remove_duplicates",
            "column": None,
            "parameters": {},
        }

        cleaned, changes = execute_operation(
            df,
            request,
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(len(changes), 1)

    def test_execute_remove_units(self):
        df = pd.DataFrame(
            {
                "weight": ["10kg", "20kg", "30kg"],
            }
        )

        request = {
            "operation": "remove_units",
            "column": "weight",
            "parameters": {
                "unit": "kg",
            },
        }

        cleaned, changes = execute_operation(
            df,
            request,
        )

        self.assertEqual(
            cleaned["weight"].tolist(),
            [10, 20, 30],
        )
        self.assertEqual(len(changes), 3)

    def test_execute_fill_numeric(self):
        df = pd.DataFrame(
            {
                "age": [20, 30, None],
            }
        )

        request = {
            "operation": "fill_numeric",
            "column": None,
            "parameters": {
                "strategy": "mean",
            },
        }

        cleaned, changes = execute_operation(
            df,
            request,
        )

        self.assertEqual(
            cleaned["age"].tolist(),
            [20, 30, 25],
        )
        self.assertEqual(len(changes), 1)

    def test_execute_operation_requires_column(self):
        df = pd.DataFrame(
            {
                "weight": ["10kg", "20kg"],
            }
        )

        request = {
            "operation": "remove_units",
            "column": None,
            "parameters": {
                "unit": "kg",
            },
        }

        with self.assertRaises(ValueError):
            execute_operation(
                df,
                request,
            )

    def test_execute_operation_rejects_unneeded_column(self):
        df = pd.DataFrame(
            {
                "name": ["A", "B"],
            }
        )

        request = {
            "operation": "remove_duplicates",
            "column": "name",
            "parameters": {},
        }

        with self.assertRaises(ValueError):
            execute_operation(
                df,
                request,
            )

    def test_normalize_instruction_whitespace(self):
        result = normalize_instruction("  Remove    duplicates  ")

        self.assertEqual(
            result,
            "Remove duplicates",
        )

    def test_normalize_instruction_synonyms(self):
        result = normalize_instruction("Replace empty numbers")

        self.assertEqual(
            result,
            "fill missing numeric values",
        )

    def test_normalize_instruction_empty(self):
        with self.assertRaises(ValueError):
            normalize_instruction("   ")


if __name__ == "__main__":
    unittest.main()
