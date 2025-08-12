# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements k-anonymity algorithms for privacy-preserving machine learning research. It compares four different k-anonymization approaches: Optimal Lattice Anonymization (OLA), Mondrian, Top-Down Greedy Anonymization (TDG), and k-NN Clustering-Based (CB) Anonymization.

## Setup and Dependencies

Use either:
- `pipenv install` followed by `pipenv shell` (preferred)
- `pip3 install -r requirements.txt`

The project uses Python 3 with key dependencies: pandas, scikit-learn, xgboost, matplotlib, seaborn, and numpy.

## Main Commands

### Running Experiments
The primary entry point is `baseline_with_repetitions.py`:

```bash
python baseline_with_repetitions.py [dataset] [classifier] [anonymization_method] [options]
```

**Datasets**: `cmc`, `mgm`, `adult`, `cahousing`
**Classifiers**: `rf` (Random Forest), `knn`, `svm`, `xgb` (XGBoost)
**Anonymization Methods**: `mondrian`, `ola`, `tdg`, `cb`

**Common Options**:
- `--start-k START_K`: Initial k value for k-anonymity (default varies)
- `--stop-k STOP_K`: Maximum k value
- `--step-k STEP_K`: Step size for k increments
- `--debug`, `-d`: Enable debugging output
- `--verbose`, `-v`: Verbose output

### Examples
```bash
# Run Random Forest on Adult dataset with Mondrian anonymization
python baseline_with_repetitions.py adult rf mondrian --start-k 2 --stop-k 100 --step-k 2

# Run OLA with suppression parameters
python baseline_with_repetitions.py adult rf ola --start-k 2 --stop-k 100 --start-s 0 --stop-s 9
```

## Architecture

### Core Components

1. **Anonymization Algorithms** (separate modules):
   - `basic_mondrian/`: Mondrian algorithm implementation
   - `clustering_based/`: k-NN clustering-based anonymization
   - `top_down_greedy/`: Top-Down Greedy anonymization
   - `elemam/`: Optimal Lattice Anonymization (OLA) implementation

2. **Data Management** (`utils/`):
   - `data.py`: Dataset reading/writing functions
   - `types.py`: Enums for datasets, classifiers, anonymization methods
   - `utility.py`: General utility functions

3. **Generalization** (`generalization/`):
   - `generalization.py`: Generalization functions (age, hierarchy-based, segmentation)
   - `hierarchy_utilities.py`: Hierarchy file reading utilities
   - `hierarchies/`: CSV files defining generalization hierarchies for each dataset

### Data Flow

1. **Data Loading**: Raw datasets loaded from `datasets/[dataset]/[dataset].csv`
2. **Baseline ML**: Original dataset performance measured with specified classifier
3. **Anonymization**: Dataset anonymized using selected algorithm and k value
4. **ML Evaluation**: Anonymized dataset performance measured
5. **Results Storage**: Results saved to `results/` with timestamp-based folder structure

### Key Data Structures

- `MLRes`: Named tuple for ML results (accuracy, precision, recall, f1_score)
- `Dataset`, `AnonMethod`, `Classifier`: Enums for type safety
- Quasi-identifier (QI) indexes and categorical flags control which columns are anonymized

### Result Organization

- `results/`: Runtime experiment results with timestamp folders
- `paper_results/`: Published paper results organized by dataset/algorithm/classifier
- `figures/`: Paper figures and visualizations

## Development Notes

- All anonymization algorithms are adapted from Qiyuan Gong's implementations, migrated from Python 2 to Python 3
- The codebase preserves non-QID attributes and target variables during anonymization
- Numeric data handling includes float support and count-based sorting
- Each experiment automatically handles train/test splits and multiple k values