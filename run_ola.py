#!/usr/bin/env python3
"""
Optimal Lattice Anonymization (OLA) k-Anonymity Algorithm Runner

This script provides a clean interface to run the OLA k-anonymity algorithm
on arbitrary datasets using the original implementation.

Usage:
    python run_ola.py input.csv output.csv --k 5 [options]

Example:
    python run_ola.py datasets/adult/adult.csv adult_ola_k5.csv --k 5 --suppression 0
"""

import argparse
import sys
import time
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from elemam.main import main as emain
from utils.data import read_data, write_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Optimal Lattice Anonymization (OLA) k-anonymity algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path for anonymized data')
    
    # k-anonymity parameters
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter')
    parser.add_argument('--suppression', '-s', type=int, default=0, 
                       help='Suppression rate (0-9, where 0=0%% and 9=90%% max suppression)')
    
    # Dataset configuration
    parser.add_argument('--dataset', choices=['adult', 'cmc', 'mgm', 'cahousing'], 
                       default='adult', help='Dataset type (affects QI/SA column selection)')
    
    # OLA-specific options
    parser.add_argument('--metric', choices=['aecs', 'dm', 'gweight', 'prec'], 
                       default='prec', help='Optimization metric for OLA')
    
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


def get_metric_name(metric):
    """Get full metric name"""
    names = {
        'aecs': 'Average Equivalence Class Size',
        'dm': 'Discernibility Metric',
        'gweight': 'Generalization Weight',
        'prec': 'Precision'
    }
    return names.get(metric, metric)


def print_configuration(args, config):
    """Print the anonymization configuration"""
    print("=" * 60)
    print("OPTIMAL LATTICE ANONYMIZATION (OLA)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Dataset: {config['name']}")
    print(f"Algorithm: OLA (Optimal Lattice Anonymization)")
    print(f"k-anonymity: {args.k}")
    print(f"Suppression rate: {args.suppression * 10}%")
    print(f"Optimization metric: {get_metric_name(args.metric)}")
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
    
    # Run OLA anonymization
    print(f"Running OLA k-anonymization (k={args.k}, suppression={args.suppression * 10}%, metric={get_metric_name(args.metric)})...")
    start_time = time.time()
    
    try:
        # Call the original OLA implementation
        anonymized_data, (ncp, runtime) = run_algorithm(
            data=data,
            k=args.k,
            qi_index=config['qi_indices'],
            sa_index=config['sa_indices'],
            suppression_rate=args.suppression,
            metric=args.metric
        )
        
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
        print("OLA k-anonymization completed successfully!")
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()