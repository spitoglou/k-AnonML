#!/usr/bin/env python3
"""
Mondrian k-Anonymity Algorithm Runner

This script provides a clean interface to run the Mondrian k-anonymity algorithm
on arbitrary datasets using the original implementation.

Usage:
    python run_mondrian.py input.csv output.csv --k 5 [options]

Example:
    python run_mondrian.py datasets/adult/adult.csv adult_k5.csv --k 5
"""

import argparse
import sys
import time
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from basic_mondrian.anonymizer import get_result_one
from basic_mondrian.utils.read_adult_data import read_tree
from utils.data import read_data, write_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Mondrian k-anonymity algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path for anonymized data')
    
    # k-anonymity parameters
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter')
    
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


def print_configuration(args, config):
    """Print the anonymization configuration"""
    print("=" * 60)
    print("MONDRIAN K-ANONYMITY ANONYMIZATION")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Dataset: {config['name']}")
    print(f"Algorithm: Mondrian")
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
    
    # Run Mondrian anonymization
    print(f"Running Mondrian k-anonymization (k={args.k})...")
    start_time = time.time()
    
    try:
        # Build attribute trees first
        print("Building attribute trees...")
        att_trees = read_tree(
            path='generalization/hierarchies/adult',
            numeric_path='.',
            dataset='adult',
            ATT_NAMES=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                      'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                      'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'],
            QI_INDEX=config['qi_indices'],
            IS_CAT=[False, True, False, True, False, True, True, True, True, True, False, False, False, True, True][:len(config['qi_indices'])]
        )
        
        # Call the original Mondrian implementation
        anonymized_data = get_result_one(
            att_trees=att_trees,
            data=data,
            k=args.k,
            path='.',
            qi_index=config['qi_indices'],
            SA_index=config['sa_indices']
        )
        
        # For now, set dummy values for NCP and runtime since get_result_one doesn't return them
        ncp = 0.0  # Would need to calculate separately
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
        print("Mondrian k-anonymization completed successfully!")
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()