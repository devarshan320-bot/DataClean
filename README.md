# DataClean

## Overview
DataClean is a data-quality analysis and safe data-cleaning tool that detects common data-quality issues, explains them, suggests controlled resolutions, previews changes, requires user approval, applies deterministic transformations, and supports re-analysis/auditing.

## Problem Statement
Working with messy CSV datasets is a ubiquitous challenge in data science and engineering. However, blindly modifying data—whether manually or through opaque AI-generated scripts—is highly risky and can lead to silent data corruption, lost evidence, and unrepeatable workflows. DataClean solves this by providing a safe, deterministic, human-in-the-loop environment for data cleaning.

## Key Features
- CSV data inspection
- Missing-value detection and cleaning
- Duplicate detection/removal
- Text inconsistency detection and standardization
- Date-format inconsistency detection
- Embedded-unit detection
- Unit standardization/conversion
- Data validation
- Data quality profiling
- Natural-language cleaning instructions
- Local AI-assisted suggestions
- Deterministic operation validation/execution
- Preview before applying changes
- Human approval before modifications
- Safe output handling
- Audit information
- Streamlit interface
- CLI interface
- Automated tests

## Safety Architecture
DataClean is built on a strict safety pipeline:
**detect → explain → classify → suggest → preview → user approval → deterministic apply → re-analyze**

**Crucially, AI does NOT directly generate or execute arbitrary pandas code.** 
The AI is strictly an interpretation and suggestion layer. AI suggestions are validated against a strict schema, and the deterministic DataClean engine performs the actual transformations. 

## Streamlit Workflow
The interactive web application follows a seamless, state-preserving workflow:
1. Upload CSV
2. Analyze Data Quality
3. Review detected issues
4. Get AI Suggestion
5. Preview Changes
6. Apply Suggestion
7. Continue resolving independent issues
8. Download cleaned data

*Note: Multiple independent issues can remain staged simultaneously. The application maintains the original uploaded data strictly separated from the working dataset.*

## Supported Deterministic Operations
DataClean performs data transformations through a strict catalog of validated operations:
- `remove_duplicates`
- `drop_missing`
- `fill_numeric`
- `fill_categorical`
- `remove_units`
- `standardize_units`
- `standardize_text`
- `date_format`

## Unit Handling
DataClean handles physical measurement units via a generalized unit registry. Rather than relying on hardcoded dataset-specific conversions, the registry provides known physical units and performs mathematically sound conversions. Mixed units within a column can be safely standardized when valid conversions are supported.

## AI Integration

- **Local First:** The AI interpretation layer uses local LLMs via Ollama when configured.
- **Role:** AI is used for semantic interpretation and resolution suggestions.
- **Authority:** Deterministic validation always remains authoritative over AI.
- **Human Review:** Ambiguous situations where the AI cannot confidently guarantee a safe resolution fall back to manual human review.

## Design Principles
- **Safety over aggressive automation**
- **Deterministic execution**
- **Human-in-the-loop**
- **Generalized detection instead of dataset-specific hardcoding**
- **Preview before modification**
- **Original data preserved**

## Testing
DataClean ensures reliability through a comprehensive test suite.
**Current Status:** 107 / 107 tests passing.

## Installation
Run the following commands in Windows PowerShell to set up the environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running the CLI
DataClean provides a robust Command Line Interface for headless operations:

**Inspect a dataset:**
```powershell
python main.py inspect data/sample.csv
```

**Clean a dataset (example):**
```powershell
python main.py clean data/sample.csv --remove-duplicates --output data/sample_cleaned.csv
```
### Cleaning options

- `--remove-duplicates`
- `--drop-missing`
- `--fill-numeric {mean,median}`
- `--fill-categorical {mode}`
- `--remove-units COLUMN:UNIT`
- `--standardize-text COLUMN:STYLE`

**Validate a dataset:**
```powershell
python main.py validate data/sample.csv --min age=0 --max marks=100 --not-null email --allowed-values city=Bengaluru,Mysuru
```

### Running the Streamlit App
To launch the interactive web interface:
```powershell
streamlit run app.py
```

## Project Structure
```
DataClean/
├── app.py                      # Streamlit application entry point
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── commands.py                 # CLI commands and suggestion orchestration
├── quality_analyzer.py         # Initial data quality issue detection
├── profiler.py                 # Data profiling
├── suggestion_resolver.py      # Resolves AI suggestions and handles ambiguous fallbacks
├── local_ai_interpreter.py     # Local AI-assisted suggestion generator
├── operation_executor.py       # Deterministic execution and safety validation
├── operations.py               # Operation schemas and metadata
├── cleaner.py                  # Core deterministic cleaning logic
├── unit_registry.py            # Physical unit definitions and conversion logic
├── benchmarks/                 # End-to-end testing datasets (e.g. messy_sales.csv)
├── tests/
│   ├── test_dataclean.py           # Core test suite
│   └── test_state_management.py    # Streamlit state progression test suite
└── README.md                   # Project documentation
```