# Enhancing Classification with Hybrid Feature Selection: A Multi-Objective Genetic Algorithm for High-Dimensional Data

This repository provides an end-to-end implementation of a **hybrid multi-objective genetic algorithm (NSGA-II)** for **feature selection** on **high-dimensional image datasets**.  
It combines **deep CNN feature extraction (DenseNet121)** with **evolutionary optimization** to select compact, discriminative feature subsets that balance **classification accuracy** and **model simplicity**.

---

## 1. Overview

This project automates the full workflow for feature selection and classification performance improvement.  
It supports multiple datasets and handles all stages of preprocessing, extraction, and optimization.

**Core workflow:**
1. Detect or generate train/validation/test splits.  
2. Extract DenseNet121 CNN features for all images.  
3. Apply NSGA-II for hybrid multi-objective feature selection.  
4. Evaluate feature subsets using classification metrics.  
5. Generate plots, reports, and summary results.

The pipeline supports **three datasets**, which should be placed under the root `data/` directory as compressed `.zip` files.

---

## 2. Requirements

| Library | Version (tested) |
|----------|------------------|
| Python | 3.8+ |
| TensorFlow | 2.x |
| scikit-learn | latest |
| NumPy | latest |
| Pillow | latest |
| DEAP | latest |
| Matplotlib | latest |
| Seaborn | latest |
| pandas | latest |

### Installation

Install all dependencies with:

```bash
pip install -r requirements.txt
```


## 3. Folder Structure

Before running any scripts, ensure your project structure looks like this:
project_root/
├── code/
│   ├── base_feature_selection.py
│   ├── classification_evaluation.py
│   ├── feature_extraction.py
│   ├── main.py
│   ├── nsga2_optimization.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── zip_files_extraction.py
│
├── data/
│   ├── dataset1.zip
│   ├── dataset2.zip
│   ├── dataset3.zip
│
├── README.md
└── LICENSE

## 4. Zip Extraction Script

The script below safely extracts all .zip files from the ./data/ folder into ./data/extracteddata/.

Save this script as:
code/zip_files_extraction.py

Then run it before feature extraction.
```python
import os
import zipfile
import hashlib

SOURCE_FOLDER = r"./data/"
DESTINATION_FOLDER = os.path.join(SOURCE_FOLDER, "extracteddata")
MAX_NAME_LEN = 50
FLATTEN_PATHS = False 

def shorten_name(name, max_len=MAX_NAME_LEN):
    """Shorten overly long names safely with an MD5 hash suffix."""
    if len(name) <= max_len:
        return name
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return name[:max_len - 9] + "_" + hash_suffix

def win_long_path(path):
    """Return a Windows-compatible long path (\\?\\ prefix)."""
    path = os.path.abspath(path)
    if not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path
    return path

def safe_extract(zip_ref, extract_path):
    """Safely extract files from a zip into the given path."""
    for member in zip_ref.infolist():
        if member.is_dir():
            continue

        orig_path = member.filename
        if FLATTEN_PATHS:
            parts = [os.path.basename(orig_path)]
        else:
            parts = [shorten_name(p) for p in orig_path.split('/') if p and p != "."]

        safe_path = os.path.join(extract_path, *parts)

        if not os.path.abspath(safe_path).startswith(os.path.abspath(extract_path)):
            print(f" Skipping unsafe path: {safe_path}")
            continue

        safe_path = win_long_path(safe_path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        with zip_ref.open(member) as source, open(safe_path, "wb") as target:
            target.write(source.read())

def extract_nested_zips(base_path):
    """Recursively extract any zip files inside base_path."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.zip'):
                nested_zip_path = os.path.join(root, file)
                nested_extract_path = root  

                try:
                    with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip:
                        safe_extract(nested_zip, nested_extract_path)
                    print(f"Extracted nested ZIP: {nested_zip_path}")

                    extract_nested_zips(nested_extract_path)

                    os.remove(nested_zip_path)
                except zipfile.BadZipFile:
                    print(f"Skipping bad ZIP file: {nested_zip_path}")

def main():
    os.makedirs(win_long_path(DESTINATION_FOLDER), exist_ok=True)

    for filename in os.listdir(SOURCE_FOLDER):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(SOURCE_FOLDER, filename)
            dataset_name = shorten_name(os.path.splitext(filename)[0])
            extract_path = os.path.join(DESTINATION_FOLDER, dataset_name)
            os.makedirs(win_long_path(extract_path), exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    safe_extract(zip_ref, extract_path)
                extract_nested_zips(extract_path)
                print(f"Extracted: {filename} → {extract_path}")

                os.remove(zip_path)
                print(f"Deleted top-level ZIP: {zip_path}")

            except zipfile.BadZipFile:
                print(f"Invalid ZIP: {filename}")

    print(f"\All zip files extracted into: {DESTINATION_FOLDER}")

if __name__ == "__main__":
    main()

```

Run this script from the project root:

```bash
python code/zip_files_extraction.py

```

## 5. Generated Folders During Execution

During execution, the following folders will be automatically generated in the project root:

| Folder                | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `data/extracteddata/` | Created by the zip extraction script; contains all unzipped dataset folders |
| `features/`           | Contains extracted CNN feature files in `.npz` format                       |
| `results/`            | Contains evaluation results, Pareto fronts, and summary plots               |

After extraction, the structure will look like this:

project_root/
├── code/
├── data/
│   ├── dataset1.zip
│   ├── dataset2.zip
│   ├── dataset3.zip
│   └── extracteddata/
│       ├── dataset1/
│       ├── dataset2/
│       ├── dataset3/
├── features/
├── results/
└── README.md


## 6. Running the Main Pipeline

After data extraction, execute the main workflow.

1. Navigate to the code folder:

```bash
cd ./code/

```

2. Run the main pipeline:

```bash
python main.py

```

## 7. Code Folder Contents
| File                           | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `base_feature_selection.py`    | Implements filter-based feature ranking and selection       |
| `classification_evaluation.py` | Evaluates model accuracy, precision, recall, and F1-score   |
| `feature_extraction.py`        | Performs DenseNet121 CNN feature extraction                 |
| `main.py`                      | Main pipeline orchestrator                                  |
| `nsga2_optimization.py`        | Implements NSGA-II multi-objective genetic algorithm        |
| `preprocessing.py`             | Handles dataset preprocessing, loading, and splitting       |
| `visualization.py`             | Generates plots and visual summaries (Matplotlib + Seaborn) |
| `zip_files_extraction.py`      | Safely extracts top-level and nested ZIP archives           |


## 8. Output Artifacts

| File/Folder                       | Description                            |
| --------------------------------- | -------------------------------------- |
| `features/*.npz`                  | Extracted CNN features and labels      |
| `results/pareto_front.png`        | NSGA-II Pareto front visualization     |
| `results/correlation_heatmap.png` | Feature correlation heatmap (Seaborn)  |
| `results/evaluation_summary.csv`  | Classification performance summary     |
| `results/selected_features.json`  | Indices of features selected by the GA |


## 9. Troubleshooting
| Issue                        | Possible Cause                 | Solution                                      |
| ---------------------------- | ------------------------------ | --------------------------------------------- |
| Path length error            | Deep folder nesting on Windows | Extract data to a shorter path like `C:\data` |
| Missing train/val/test split | Dataset not pre-split          | Automatic stratified split will be generated  |
| Corrupted image files        | Damaged image data             | Automatically skipped and logged              |
| Memory issue                 | Batch size too large           | Reduce `batch_size` in `main.py`              |


