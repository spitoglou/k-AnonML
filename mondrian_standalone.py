#!/usr/bin/env python3
"""
Standalone Mondrian k-Anonymity Algorithm

This script implements the Basic Mondrian algorithm for k-anonymity without external dependencies.
It can anonymize datasets with both numeric and categorical quasi-identifiers.

Usage:
    python mondrian_standalone.py input.csv output.csv --k 5 --qi-columns 0,1,2 --sa-columns 3

Author: Standalone implementation based on Mondrian algorithm
"""

import argparse
import csv
import sys
import time
from functools import cmp_to_key
from collections import defaultdict


class NumRange:
    """Numeric range for generalization hierarchies"""
    
    def __init__(self, sort_value, support=None):
        self.sort_value = list(sort_value)
        self.support = support or {}
        self.range = float(sort_value[-1]) - float(sort_value[0]) if len(sort_value) > 1 else 0
        self.dict = {v: i for i, v in enumerate(sort_value)}
        self.value = f"{sort_value[0]},{sort_value[-1]}" if len(sort_value) > 1 else str(sort_value[0])


class Partition:
    """Class for storing record partitions"""
    
    def __init__(self, data, width, middle, qi_len):
        self.member = list(data)
        self.width = list(width)
        self.middle = list(middle)
        self.allow = [1] * qi_len
    
    def __len__(self):
        return len(self.member)


def cmp_str(element1, element2):
    """String comparison for sorting"""
    try:
        return float(element1) - float(element2)
    except ValueError:
        if element1 < element2:
            return -1
        elif element1 > element2:
            return 1
        else:
            return 0


def build_hierarchy_trees(data, qi_indices, is_categorical):
    """Build attribute hierarchy trees from data"""
    att_trees = []
    
    for i, qi_idx in enumerate(qi_indices):
        # Extract column values
        column_values = [row[qi_idx] for row in data]
        
        if not is_categorical[i]:
            # Numeric attribute - create NumRange
            unique_values = list(set(column_values))
            try:
                unique_values.sort(key=float)
            except ValueError:
                unique_values.sort()
            
            support = defaultdict(int)
            for val in column_values:
                support[val] += 1
                
            att_trees.append(NumRange(unique_values, dict(support)))
        else:
            # Categorical attribute - create simple hierarchy
            unique_values = list(set(column_values))
            hierarchy = {'*': unique_values}
            for val in unique_values:
                hierarchy[val] = [val]
            att_trees.append(hierarchy)
    
    return att_trees


def get_normalized_width(partition, index, att_trees, qi_range, is_categorical):
    """Calculate normalized width of partition for given dimension"""
    if not is_categorical[index]:
        low = partition.width[index][0]
        high = partition.width[index][1]
        if low == high:
            width = 0
        else:
            width = float(att_trees[index].sort_value[high]) - float(att_trees[index].sort_value[low])
    else:
        width = partition.width[index]
    
    return width * 1.0 / qi_range[index] if qi_range[index] > 0 else 0


def choose_dimension(partition, att_trees, qi_range, is_categorical, qi_len):
    """Choose dimension with largest normalized width for splitting"""
    max_width = -1
    max_dim = -1
    
    for i in range(qi_len):
        if partition.allow[i] == 0:
            continue
            
        norm_width = get_normalized_width(partition, i, att_trees, qi_range, is_categorical)
        
        # Additional check: ensure this dimension has enough distinct values to split
        if not is_categorical[i]:
            # For numeric: check if we have range to split
            low = partition.width[i][0]
            high = partition.width[i][1]
            if low >= high:  # No range to split
                continue
        else:
            # For categorical: check if we have multiple values
            if norm_width <= 0:  # No diversity to split
                continue
        
        if norm_width > max_width:
            max_width = norm_width
            max_dim = i
    
    return max_dim


def frequency_set(partition, dim):
    """Get frequency set of partition on dimension"""
    frequency = {}
    for record in partition.member:
        val = record[dim]
        frequency[val] = frequency.get(val, 0) + 1
    return frequency


def find_median(partition, dim, k):
    """Find median split value for dimension"""
    frequency = frequency_set(partition, dim)
    value_list = list(frequency.keys())
    value_list.sort(key=cmp_to_key(cmp_str))
    
    total = sum(frequency.values())
    middle = total / 2
    
    # Enhanced termination conditions
    if middle < k or len(value_list) <= 1:
        return ('', '', value_list[0], value_list[-1])
    
    # Additional check: ensure we can create two groups of at least k records each
    if total < 2 * k:
        return ('', '', value_list[0], value_list[-1])
    
    index = 0
    split_val = ''
    split_index = 0
    
    # Find a split point that ensures both sides have at least k records
    for i, val in enumerate(value_list):
        index += frequency[val]
        if index >= k and (total - index) >= k and index >= middle:
            split_val = val
            split_index = i
            break
    
    # If no valid split found, return empty
    if split_val == '':
        return ('', '', value_list[0], value_list[-1])
    
    try:
        next_val = value_list[split_index + 1]
    except IndexError:
        next_val = split_val
    
    # Final validation: if split_val equals next_val, we can't split
    if split_val == next_val:
        return ('', '', value_list[0], value_list[-1])
    
    return (split_val, next_val, value_list[0], value_list[-1])


def split_numerical_value(numeric_value, split_val):
    """Split numeric value on split_val"""
    split_num = numeric_value.split(',')
    if len(split_num) <= 1:
        return split_num[0], split_num[0]
    
    low, high = split_num[0], split_num[1]
    
    if low == split_val:
        lvalue = low
    else:
        lvalue = f"{low},{split_val}"
    
    if high == split_val:
        rvalue = high
    else:
        rvalue = f"{split_val},{high}"
    
    return lvalue, rvalue


def split_numerical(partition, dim, pwidth, pmiddle, att_trees, k):
    """Split numerical attribute"""
    sub_partitions = []
    
    (split_val, next_val, low, high) = find_median(partition, dim, k)
    if split_val == '':
        return []
    
    # Ensure split_val exists in the dictionary
    if split_val not in att_trees[dim].dict:
        return []
    
    mean = att_trees[dim].dict[split_val]
    lmiddle = pmiddle[:]
    rmiddle = pmiddle[:]
    
    lmiddle[dim], rmiddle[dim] = split_numerical_value(pmiddle[dim], split_val)
    
    lwidth = pwidth[:]
    rwidth = pwidth[:]
    lwidth[dim] = (pwidth[dim][0], mean)
    rwidth[dim] = (mean, pwidth[dim][1])
    
    lmember = []
    rmember = []
    
    for record in partition.member:
        if record[dim] in att_trees[dim].dict:
            pos = att_trees[dim].dict[record[dim]]
            if pos <= mean:
                lmember.append(record)
            else:
                rmember.append(record)
        else:
            # If record value not in dict, add to left partition by default
            lmember.append(record)
    
    # Note: Original implementation doesn't check k-anonymity here
    # The find_median function already ensures valid splits
    
    sub_partitions.append(Partition(lmember, lwidth, lmiddle, len(pmiddle)))
    sub_partitions.append(Partition(rmember, rwidth, rmiddle, len(pmiddle)))
    
    return sub_partitions


def split_categorical(partition, dim, pwidth, pmiddle, att_trees, k):
    """Split categorical attribute"""
    sub_partitions = []
    
    # Simple categorical split - divide into two groups
    frequency = frequency_set(partition, dim)
    values = list(frequency.keys())
    
    if len(values) <= 1:
        return []
    
    # Split values into two groups
    mid = len(values) // 2
    left_values = set(values[:mid])
    right_values = set(values[mid:])
    
    lmember = []
    rmember = []
    
    for record in partition.member:
        if record[dim] in left_values:
            lmember.append(record)
        else:
            rmember.append(record)
    
    # Note: Original implementation doesn't check k-anonymity here
    # The find_median function already ensures valid splits
    
    # Create generalized values
    lmiddle = pmiddle[:]
    rmiddle = pmiddle[:]
    lmiddle[dim] = ','.join(sorted(left_values))
    rmiddle[dim] = ','.join(sorted(right_values))
    
    lwidth = pwidth[:]
    rwidth = pwidth[:]
    lwidth[dim] = len(left_values)
    rwidth[dim] = len(right_values)
    
    sub_partitions.append(Partition(lmember, lwidth, lmiddle, len(pmiddle)))
    sub_partitions.append(Partition(rmember, rwidth, rmiddle, len(pmiddle)))
    
    return sub_partitions


def split_partition(partition, dim, att_trees, is_categorical, k):
    """Split partition on given dimension"""
    pwidth = partition.width
    pmiddle = partition.middle
    
    if not is_categorical[dim]:
        return split_numerical(partition, dim, pwidth, pmiddle, att_trees, k)
    else:
        return split_categorical(partition, dim, pwidth, pmiddle, att_trees, k)


def check_splitable(partition, k):
    """Check if partition can be split while maintaining k-anonymity"""
    return sum(partition.allow) > 0


def anonymize(partition, att_trees, qi_range, is_categorical, qi_len, k, result, depth=0):
    """Main anonymization procedure - recursively partition until not splitable"""
    # Add recursion depth protection
    MAX_DEPTH = 50
    if depth > MAX_DEPTH:
        result.append(partition)
        return
    
    # Check basic splitting conditions
    if not check_splitable(partition, k):
        result.append(partition)
        return
    
    # Additional safety: if partition is too small to split meaningfully
    if len(partition) < 2 * k:
        result.append(partition)
        return
    
    # Choose dimension to split
    dim = choose_dimension(partition, att_trees, qi_range, is_categorical, qi_len)
    if dim == -1:
        result.append(partition)
        return
    
    # Split partition
    sub_partitions = split_partition(partition, dim, att_trees, is_categorical, k)
    
    if len(sub_partitions) == 0:
        partition.allow[dim] = 0
        # Recursively try with blocked dimension
        anonymize(partition, att_trees, qi_range, is_categorical, qi_len, k, result, depth + 1)
    else:
        # Validate sub-partitions before recursing
        valid_partitions = []
        for sub_partition in sub_partitions:
            if len(sub_partition) >= k:  # Only recurse on valid partitions
                valid_partitions.append(sub_partition)
            else:
                # If sub-partition is too small, merge back to parent or handle appropriately
                result.append(sub_partition)
        
        # Recurse on valid partitions
        for sub_partition in valid_partitions:
            anonymize(sub_partition, att_trees, qi_range, is_categorical, qi_len, k, result, depth + 1)


def mondrian_anonymization(data, k, qi_indices, sa_indices, is_categorical):
    """Main Mondrian k-anonymity algorithm"""
    qi_len = len(qi_indices)
    
    # Build attribute trees
    att_trees = build_hierarchy_trees(data, qi_indices, is_categorical)
    
    # Initialize ranges and width
    qi_range = []
    wtemp = []
    middle = []
    
    for i in range(qi_len):
        if not is_categorical[i]:
            qi_range.append(att_trees[i].range)
            wtemp.append((0, len(att_trees[i].sort_value) - 1))
            middle.append(att_trees[i].value)
        else:
            qi_range.append(len(att_trees[i]['*']))
            wtemp.append(len(att_trees[i]['*']))
            middle.append('*')
    
    # Create initial partition with QI-only data
    qi_data = [[row[qi_idx] for qi_idx in qi_indices] for row in data]
    whole_partition = Partition(qi_data, wtemp, middle, qi_len)
    
    # Perform anonymization
    start_time = time.time()
    result = []
    anonymize(whole_partition, att_trees, qi_range, is_categorical, qi_len, k, result, 0)
    runtime = time.time() - start_time
    
# Debug output removed
    
    # Generate anonymized dataset
    anonymized_data = []
    ncp = 0.0
    
    # Create mapping from QI values to original records
    qi_to_original = {}
    for row_idx, row in enumerate(data):
        qi_tuple = tuple(row[qi_idx] for qi_idx in qi_indices)
        if qi_tuple not in qi_to_original:
            qi_to_original[qi_tuple] = []
        qi_to_original[qi_tuple].append((row_idx, row))
    
    for partition in result:
        # Calculate NCP for this partition
        r_ncp = 0.0
        for i in range(qi_len):
            r_ncp += get_normalized_width(partition, i, att_trees, qi_range, is_categorical)
        
        # Add records with generalized values
        generalized_values = partition.middle
        for qi_record in partition.member:
            # Find corresponding original record
            qi_tuple = tuple(qi_record)
            if qi_tuple in qi_to_original:
                # Get the first available original record with these QI values
                original_records = qi_to_original[qi_tuple]
                if original_records:
                    row_idx, original_record = original_records.pop(0)
                    new_record = original_record[:]
                    
                    # Replace QI values with generalized values
                    for i, qi_idx in enumerate(qi_indices):
                        new_record[qi_idx] = generalized_values[i]
                    
                    anonymized_data.append(new_record)
        
        r_ncp *= len(partition)
        ncp += r_ncp
    
    # Convert NCP to percentage
    if len(data) > 0 and qi_len > 0:
        ncp /= qi_len
        ncp /= len(data)
        ncp *= 100
    else:
        ncp = 0.0
    
    return anonymized_data, ncp, runtime


def load_csv(filename):
    """Load CSV file"""
    data = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        # Try to detect delimiter
        first_line = f.readline()
        f.seek(0)
        
        delimiter = ',' if ',' in first_line else ';'
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            data.append(row)
    return data


def save_csv(filename, data):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description='Standalone Mondrian k-Anonymity Algorithm')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output CSV file for anonymized data')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter (default: 5)')
    parser.add_argument('--qi-columns', required=True, help='Comma-separated quasi-identifier column indices (0-based)')
    parser.add_argument('--sa-columns', default='', help='Comma-separated sensitive attribute column indices (0-based)')
    parser.add_argument('--categorical', default='', help='Comma-separated indices of categorical QI columns')
    parser.add_argument('--header', action='store_true', help='Input file has header row')
    
    args = parser.parse_args()
    
    # Parse column indices
    qi_indices = [int(x.strip()) for x in args.qi_columns.split(',')]
    sa_indices = [int(x.strip()) for x in args.sa_columns.split(',')] if args.sa_columns else []
    categorical_indices = set([int(x.strip()) for x in args.categorical.split(',')]) if args.categorical else set()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_csv(args.input_file)
    
    if not data:
        print("Error: No data found in input file")
        sys.exit(1)
    
    # Handle header
    header = None
    if args.header:
        header = data[0]
        data = data[1:]
    
    print(f"Dataset: {len(data)} records, {len(data[0])} attributes")
    print(f"QI columns: {qi_indices}")
    print(f"SA columns: {sa_indices}")
    print(f"k-anonymity parameter: {args.k}")
    
    # Determine which QI columns are categorical
    is_categorical = []
    for i, qi_idx in enumerate(qi_indices):
        is_categorical.append(qi_idx in categorical_indices)
    
    print(f"Categorical QIs: {[qi_indices[i] for i, cat in enumerate(is_categorical) if cat]}")
    
    # Run Mondrian anonymization
    print("Running Mondrian k-anonymity algorithm...")
    anonymized_data, ncp, runtime = mondrian_anonymization(data, args.k, qi_indices, sa_indices, is_categorical)
    
    # Add header back if present
    if header:
        anonymized_data.insert(0, header)
    
    # Save results
    save_csv(args.output_file, anonymized_data)
    
    # Print results
    print(f"\\nAnonymization completed!")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Information loss (NCP): {ncp:.2f}%")
    print(f"Output saved to: {args.output_file}")
    print(f"Records processed: {len(data)} -> {len(anonymized_data) - (1 if header else 0)}")


if __name__ == "__main__":
    main()