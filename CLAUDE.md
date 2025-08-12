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

## Standalone Algorithm Implementations

**NEW: Self-contained k-anonymity algorithm scripts have been created for production use:**

### Available Standalone Scripts
- `mondrian_standalone.py`: Mondrian k-anonymity algorithm
- `tdg_standalone.py`: Top-Down Greedy (TDG) algorithm  
- `cb_standalone.py`: Clustering-Based (CB) algorithms (k-NN, k-Member, OKA)
- `ola_standalone.py`: Optimal Lattice Anonymization (OLA) - **REQUIRES FIXES**

### Usage Pattern
```bash
python [script_name] input.csv output.csv --k [value] --qi-columns [indices] [options]
```

### Common Parameters
- `--k`: k-anonymity parameter (default: 5)
- `--qi-columns`: Comma-separated QI column indices (required)
- `--categorical`: Comma-separated categorical QI column indices
- `--header`: Input file has header row
- `--seed`: Random seed for reproducible results

### Examples
```bash
# Mondrian algorithm
python mondrian_standalone.py adult_subset_1000.csv output.csv --k 5 --qi-columns 0,1,3,4,5,6,7 --categorical 1,3,4,5,6,7 --header

# TDG algorithm
python tdg_standalone.py adult_subset_1000.csv output.csv --k 5 --qi-columns 0,1,3,4,5,6,7 --categorical 1,3,4,5,6,7 --header

# CB algorithms
python cb_standalone.py adult_subset_1000.csv output.csv --k 5 --qi-columns 0,1,3,4,5,6,7 --categorical 1,3,4,5,6,7 --header --cb-alg [knn|kmember|oka]
```

### Performance Results (1000-record Adult Dataset, k=5)
| Algorithm | Runtime | Information Loss (NCP) | Quality Rank |
|-----------|---------|----------------------|--------------|
| Mondrian | 0.01s | 29.67% | 3rd |
| TDG | 0.35s | 28.25% | 2nd |
| CB k-NN | 0.24s | 36.94% | 5th |
| CB k-Member | 5.11s | 23.64% | 1st (best) |
| CB OKA | 2.22s | 32.29% | 4th |

## Testing and Comparison Tools

### Algorithm Comparison Scripts
- `compare_algorithms.py`: Comprehensive comparison framework (baseline script integration)
- `direct_comparison.py`: Direct function call comparison approach
- `adult_subset_1000.csv`: Test dataset with 1000 records for validation

### Test Dataset Generation
```bash
# Create test subset from adult dataset
head -n 1001 datasets/adult/adult.csv > adult_subset_1000.csv
```

## Current Status and Issues

### ✅ Completed Work
1. **Mondrian Standalone**: Fully functional, performance-optimized
2. **TDG Standalone**: Fully functional, maintains algorithmic integrity
3. **CB Standalone**: All 3 variants (k-NN, k-Member, OKA) implemented and working
4. **Documentation**: Comprehensive algorithm analysis files created
5. **Testing**: Comparison framework and validation completed

### ⚠️ Known Issues and Pending Work

#### OLA Implementation Issues
- **Status**: Requires major architectural rewrite
- **Problem**: Complex hierarchy indexing errors (`string used as list index`)
- **Impact**: Non-functional for practical use
- **Recommendation**: Significant development effort needed or use alternative algorithms

#### Integration with Original Codebase
- **Status**: Original implementations have dependency and data format constraints
- **Problem**: Complex attribute tree requirements, specific data formats
- **Impact**: Standalone versions are more reliable for production use
- **Recommendation**: Use standalone implementations for new projects

#### Performance Considerations
- **CB k-NN**: Optimized for speed with candidate sampling (may reduce clustering quality)
- **Large Datasets**: Test performance with datasets >10k records
- **Memory Usage**: Monitor memory consumption for clustering algorithms

### 🎯 Recommendations for Future Development

#### Immediate Priorities
1. **Fix OLA Implementation**: Rewrite hierarchy handling logic
2. **Performance Testing**: Validate algorithms on larger datasets
3. **Integration Testing**: Test with actual ML workflows

#### Long-term Enhancements
1. **GUI Interface**: Create user-friendly interface for algorithm selection
2. **Batch Processing**: Add support for multiple file processing
3. **Advanced Metrics**: Implement additional utility metrics beyond NCP
4. **Cloud Deployment**: Package for cloud/container deployment

## Development Notes

- All anonymization algorithms are adapted from Qiyuan Gong's implementations, migrated from Python 2 to Python 3
- The codebase preserves non-QID attributes and target variables during anonymization
- Numeric data handling includes float support and count-based sorting
- Each experiment automatically handles train/test splits and multiple k values
- **NEW**: Standalone implementations provide production-ready alternatives with improved error handling and self-contained execution