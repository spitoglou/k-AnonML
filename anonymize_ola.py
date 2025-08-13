#!/usr/bin/env python3
"""
Simple OLA (Optimal Lattice Anonymization) k-Anonymity Script

Usage: python anonymize_ola.py --dataset adult --k 5

Note: OLA implementation has known issues and requires fixes.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required modules
from utils.types import Dataset, AnonMethod
from elemam.main import main as emain
from generalization.generalization import age, hierarchy, l1sub
from utils.data import read_raw


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run OLA k-anonymity algorithm')
    parser.add_argument('--dataset', choices=['adult', 'cmc', 'mgm', 'cahousing'], 
                       default='adult', help='Dataset to anonymize')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter')
    parser.add_argument('--suppression', type=int, default=0, help='Suppression rate (0-9)')
    parser.add_argument('--metric', choices=['aecs', 'dm', 'gweight', 'prec'], 
                       default='prec', help='Optimization metric')
    parser.add_argument('--output', help='Output file (default: [dataset]_ola_k[k].csv)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()


def get_dataset_config(dataset_name):
    """Get dataset configuration"""
    configs = {
        'adult': {
            'path': 'datasets/adult',
            'hierarchies_path': 'generalization/hierarchies/adult',
            'qi_index': [1, 2, 3, 4, 5, 6, 7, 8],
            'sa_index': [0, 9],
            'is_cat': [True, False, True, True, True, True, True, True]
        }
    }
    return configs.get(dataset_name, configs['adult'])


def main():
    args = parse_arguments()
    config = get_dataset_config(args.dataset)
    
    # Setup output file
    if not args.output:
        args.output = f"{args.dataset}_ola_k{args.k}_s{args.suppression}.csv"
    
    print("=" * 60)
    print("OLA (OPTIMAL LATTICE ANONYMIZATION) K-ANONYMITY")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"k-anonymity: {args.k}")
    print(f"Suppression rate: {args.suppression * 10}%")
    print(f"Metric: {args.metric}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    print("WARNING: OLA implementation has known issues and may fail.")
    print("=" * 60)
    
    try:
        # Read raw data
        print("Loading dataset...")
        raw_data, header = read_raw(
            path=config['path'],
            numeric_path='.',
            dataset=args.dataset,
            qi_index=config['qi_index'],
            is_cat=config['is_cat']
        )
        print(f"Loaded {len(raw_data)} records with {len(header)} attributes")
        
        # Setup generalization strategies (from baseline script)
        gen_path = 'generalization/hierarchies/'
        gen_strat = [
            l1sub, age, l1sub,
            hierarchy(gen_path + args.dataset, 'marital-status'),
            hierarchy(gen_path + args.dataset, 'education'),
            hierarchy(gen_path + args.dataset, 'native-country'),
            hierarchy(gen_path + args.dataset, 'workclass'),
            hierarchy(gen_path + args.dataset, 'occupation')
        ]
        max_gen_level = [1, 4, 1, 2, 3, 2, 2, 2]
        
        print("Running OLA k-anonymization...")
        print("Note: This may fail due to known OLA implementation issues.")
        start_time = time.time()
        
        # Call OLA main function
        anonymized_data, gen_level_array = emain(
            raw_data=raw_data,
            kanon=args.k,
            gen_strat=gen_strat,
            max_gen_level=max_gen_level,
            qi_index=config['qi_index'],
            metric=args.metric,
            res_folder='.',
            suppression_rate=args.suppression
        )
        
        runtime = time.time() - start_time
        
        # Save results
        print("Saving anonymized data...")
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(anonymized_data)
        
        print("=" * 60)
        print("ANONYMIZATION COMPLETED")
        print("=" * 60)
        print(f"Original records: {len(raw_data)}")
        print(f"Anonymized records: {len(anonymized_data)}")
        print(f"Runtime: {runtime:.3f} seconds")
        print(f"Output saved to: {args.output}")
        print("=" * 60)
        print("Success!")
        
    except Exception as e:
        print("=" * 60)
        print("OLA ANONYMIZATION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nThis is a known issue with the OLA implementation.")
        print("The OLA algorithm requires significant fixes to work properly.")
        print("Consider using Mondrian, TDG, or CB algorithms instead.")
        print("=" * 60)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()