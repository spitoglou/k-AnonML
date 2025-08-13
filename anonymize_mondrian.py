#!/usr/bin/env python3
"""
Simple Mondrian k-Anonymity Script

Usage: python anonymize_mondrian.py --dataset adult --k 5
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
from basic_mondrian.anonymizer import get_result_one
from basic_mondrian.utils.read_adult_data import read_tree
from utils.data import read_raw


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Mondrian k-anonymity algorithm')
    parser.add_argument('--dataset', choices=['adult', 'cmc', 'mgm', 'cahousing'], 
                       default='adult', help='Dataset to anonymize')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter')
    parser.add_argument('--output', help='Output file (default: [dataset]_mondrian_k[k].csv)')
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
        args.output = f"{args.dataset}_mondrian_k{args.k}.csv"
    
    print("=" * 60)
    print("MONDRIAN K-ANONYMITY ANONYMIZATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"k-anonymity: {args.k}")
    print(f"Output file: {args.output}")
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
        
        # Build attribute trees
        print("Building attribute trees...")
        att_trees = read_tree(
            path=config['hierarchies_path'],
            numeric_path='.',
            dataset=args.dataset,
            ATT_NAMES=header,
            QI_INDEX=config['qi_index'],
            IS_CAT=config['is_cat'][:len(config['qi_index'])]
        )
        print(f"Built {len(att_trees)} attribute trees")
        
        # Run anonymization
        print(f"Running Mondrian k-anonymization (k={args.k})...")
        start_time = time.time()
        
        anonymized_data = get_result_one(
            att_trees=att_trees,
            data=raw_data,
            k=args.k,
            path='.',
            qi_index=config['qi_index'],
            SA_index=config['sa_index']
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
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()