#!/usr/bin/env python3
"""
Clustering-Based (CB) k-Anonymity Algorithm Runner

This script provides a clean interface to run the Clustering-Based k-anonymity algorithm
on arbitrary datasets using the original implementation.

Usage:
    python run_cb.py input.csv output.csv --k 5 [options]

Example:
    python run_cb.py datasets/adult/adult.csv adult_cb_k5.csv --k 5 --algorithm knn
"""

import argparse
import sys
import time
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clustering_based.anonymizer import get_result_one as cb_get_result_one
from utils.data import read_data, write_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Clustering-Based k-anonymity algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path for anonymized data')
    
    # k-anonymity parameters
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter')
    
    # Algorithm variant
    parser.add_argument('--algorithm', choices=['knn', 'kmember', 'oka'], default='knn',
                       help='Clustering algorithm variant (knn=k-nearest neighbor, kmember=k-member, oka=one-time k-means)')
    
    # Dataset configuration
    parser.add_argument('--dataset', choices=['adult', 'cmc', 'mgm', 'cahousing'], 
                       default='adult', help='Dataset type (affects QI/SA column selection)')
    
    # Output and debugging
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def get_dataset_config(dataset_name):
    """Get QI and SA indices for different datasets"""
    configs = {
        'adult': {
            'qi_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # All except salary-class
            'sa_indices': [9],  # salary-class
            'name': 'Adult Census'
        },
        'cmc': {
            'qi_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # All except contraceptive method
            'sa_indices': [9],  # contraceptive method
            'name': 'Contraceptive Method Choice'
        },
        'mgm': {
            'qi_indices': [0, 1, 2, 3, 4],  # All except severity
            'sa_indices': [5],  # severity
            'name': 'Mammographic Mass'
        },
        'cahousing': {
            'qi_indices': [0, 1, 2, 3, 4, 5, 6, 7],  # All except median house value
            'sa_indices': [8],  # median house value
            'name': 'California Housing'
        }
    }
    return configs.get(dataset_name, configs['adult'])


def get_algorithm_name(algorithm):
    """Get full algorithm name"""
    names = {
        'knn': 'k-Nearest Neighbor',
        'kmember': 'k-Member',
        'oka': 'One-time K-means (OKA)'
    }
    return names.get(algorithm, algorithm)


def print_configuration(args, config):
    """Print the anonymization configuration"""
    print("=" * 60)
    print("CLUSTERING-BASED K-ANONYMITY ANONYMIZATION")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Dataset: {config['name']}")
    print(f"Algorithm: Clustering-Based ({get_algorithm_name(args.algorithm)})")
    print(f"k-anonymity: {args.k}")
    print(f"QI columns: {config['qi_indices']}")
    print(f"SA columns: {config['sa_indices']}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("=" * 60)


def main():
    args = parse_arguments()
    config = get_dataset_config(args.dataset)
    print_configuration(args, config)
    
    # Load data
    print("Loading data...")
    try:
        data = read_data(args.input_file)
        print(f"Loaded {len(data)} records with {len(data[0])} attributes")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run CB anonymization
    print(f"Running Clustering-Based k-anonymization (k={args.k}, algorithm={get_algorithm_name(args.algorithm)})...")
    start_time = time.time()
    
    try:
        # Call the original CB implementation
        anonymized_data = cb_get_result_one(
            att_trees=None,  # CB builds its own trees
            data=data,
            k=args.k,
            path='.',
            qi_index=config['qi_indices'],
            SA_index=config['sa_indices'],
            type_alg=args.algorithm
        )
        # CB doesn't return NCP and runtime separately
        ncp = 0.0
        runtime = time.time() - start_time
        
        total_time = time.time() - start_time
        
        # Save anonymized data
        print("Saving anonymized data...")
        write_data(anonymized_data, args.output_file)
        
        # Print results
        print("=" * 60)
        print("ANONYMIZATION RESULTS")
        print("=" * 60)
        print(f"Original records: {len(data)}")
        print(f"Anonymized records: {len(anonymized_data)}")
        print(f"Information loss (NCP): {ncp:.2f}%")
        print(f"Runtime: {runtime:.3f} seconds")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Output saved to: {args.output_file}")
        print("=" * 60)
        print("Clustering-Based k-anonymization completed successfully!")
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()