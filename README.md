# ML_projects

This repository contains a collection of machine learning and data science projects 

## Projects Overview

- **LocationParser_project**: Location Extraction from Text
  - Reads a text file and extracts location mentions using simple NLP.
  - Files: `simpleFileReader.py`, `nlpReader.py`, example inputs in `example.txt`, `example2.txt`.

- **Melbourne_project**: Melbourne Housing Data Exploration
  - Exploratory analysis and interpretation of the Melbourne housing dataset.
  - Files: `dataInterpretation.py`, dataset `melb_data.csv`.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd ML_projects
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Each project has its own scripts. Minimal examples are below.

### Running the projects

- LocationParser_project
  - Simple file reader (prints contents):
    ```bash
    python LocationParser_project/simpleFileReader.py LocationParser_project/example.txt
    ```
  - NLP reader (extracts locations; writes to `nlp_output.txt`):
    ```bash
    python LocationParser_project/nlpReader.py LocationParser_project/example.txt
    ```

- Melbourne_project
  - Data interpretation over the bundled CSV:
    ```bash
    python Melbourne_project/dataInterpretation.py
    ```
    The script expects `Melbourne_project/melb_data.csv` to be present.

---

For questions or contributions, please open an issue or pull request.
