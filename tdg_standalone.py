#!/usr/bin/env python3
"""
Standalone Top-Down Greedy (TDG) k-Anonymity Algorithm

This script implements the Top-Down Greedy algorithm for k-anonymity without external dependencies.
It uses distance-based binary partitioning to create natural clusters around centroid pairs.

Usage:
    python tdg_standalone.py input.csv output.csv --k 5 --qi-columns 0,1,2 --sa-columns 3

Author: Standalone implementation based on Top-Down Greedy algorithm
"""

import argparse
import csv
import operator
import random
import sys
import time
from collections import defaultdict
from functools import cmp_to_key


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
    
    def __init__(self, data, middle):
        self.can_split = True
        self.member = list(data)
        self.middle = list(middle)
    
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


def get_num_list_from_str(stemp):
    """Convert string to numeric list"""
    try:
        float(stemp)
        return [stemp]
    except ValueError:
        return stemp.split(',')


def build_hierarchy_trees(data, qi_indices, is_categorical):
    """Build attribute hierarchy trees from data"""
    att_trees = []
    
    for i, qi_idx in enumerate(qi_indices):
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


def NCP(record, att_trees, qi_range, is_categorical, qi_len):
    """Compute Normalized Certainty Penalty of record"""
    record_ncp = 0.0
    for i in range(qi_len):
        if not is_categorical[i]:
            value_ncp = 0
            try:
                float(record[i])
            except ValueError:
                split_number = record[i].split(',')
                value_ncp = float(split_number[1]) - float(split_number[0])
            value_ncp = value_ncp * 1.0 / qi_range[i] if qi_range[i] > 0 else 0
            record_ncp += value_ncp
        else:
            # For categorical, use simple generalization cost
            if record[i] == '*':
                record_ncp += 1.0
            else:
                record_ncp += 0.0
    return record_ncp


def NCP_dis(record1, record2, att_trees, qi_range, is_categorical, qi_len):
    """Use NCP of generalization of record1 and record2 as distance"""
    mid = middle_record(record1, record2, att_trees, is_categorical, qi_len)
    return NCP(mid, att_trees, qi_range, is_categorical, qi_len), mid


def NCP_dis_merge(partition, addition_set, att_trees, qi_range, is_categorical, qi_len):
    """Merge addition_set to current partition and compute NCP"""
    mid = middle_group(addition_set, att_trees, is_categorical, qi_len)
    mid = middle_record(mid, partition.middle, att_trees, is_categorical, qi_len)
    return (len(addition_set) + len(partition)) * NCP(mid, att_trees, qi_range, is_categorical, qi_len), mid


def middle_record(record1, record2, att_trees, is_categorical, qi_len):
    """Get generalization result of record1 and record2"""
    mid = []
    for i in range(qi_len):
        if not is_categorical[i]:
            split_number = []
            split_number.extend(get_num_list_from_str(record1[i]))
            split_number.extend(get_num_list_from_str(record2[i]))
            split_number.sort(key=cmp_to_key(cmp_str))
            # Avoid 2,2 problem
            if split_number[0] == split_number[-1]:
                mid.append(split_number[0])
            else:
                mid.append(split_number[0] + ',' + split_number[-1])
        else:
            mid.append(LCA(record1[i], record2[i], i, att_trees))
    return mid


def middle_group(group_set, att_trees, is_categorical, qi_len):
    """Get generalization result of the group"""
    len_group_set = len(group_set)
    mid = group_set[0]
    for i in range(1, len_group_set):
        mid = middle_record(mid, group_set[i], att_trees, is_categorical, qi_len)
    return mid


def LCA(u, v, index, att_trees):
    """Get lowest common ancestor for categorical attributes"""
    if u == v:
        return u
    # Simple LCA for categorical - return most general value
    return '*'


def get_pair(partition, att_trees, qi_range, is_categorical, qi_len, rounds=3):
    """Get max distance pair using heuristic method"""
    len_partition = len(partition)
    if len_partition < 2:
        return (0, 0)
    
    for i in range(rounds):
        if i == 0:
            u = random.randrange(len_partition)
        else:
            u = v
        max_dis = -1
        max_index = 0
        for j in range(len_partition):
            if j != u:
                rncp, _ = NCP_dis(partition.member[j], partition.member[u], 
                                att_trees, qi_range, is_categorical, qi_len)
                if rncp > max_dis:
                    max_dis = rncp
                    max_index = j
        v = max_index
    return (u, v)


def distribute_record(u, v, partition, att_trees, qi_range, is_categorical, qi_len):
    """Distribute records based on NCP distance"""
    record_u = partition.member[u][:]
    record_v = partition.member[v][:]
    u_partition = [record_u]
    v_partition = [record_v]
    remain_records = [item for index, item in enumerate(partition.member) 
                     if index not in {u, v}]
    
    for record in remain_records:
        u_dis, _ = NCP_dis(record_u, record, att_trees, qi_range, is_categorical, qi_len)
        v_dis, _ = NCP_dis(record_v, record, att_trees, qi_range, is_categorical, qi_len)
        if u_dis > v_dis:
            v_partition.append(record)
        else:
            u_partition.append(record)
    
    return [Partition(u_partition, middle_group(u_partition, att_trees, is_categorical, qi_len)),
            Partition(v_partition, middle_group(v_partition, att_trees, is_categorical, qi_len))]


def balance(sub_partitions, index, k, att_trees, qi_range, is_categorical, qi_len):
    """Balance partitions to ensure k-anonymity"""
    less = sub_partitions.pop(index)
    more = sub_partitions.pop()
    all_length = len(less) + len(more)
    require = k - len(less)
    
    # First method: move some records
    dist = {}
    for i, record in enumerate(more.member):
        dist[i], _ = NCP_dis(less.middle, record, att_trees, qi_range, is_categorical, qi_len)

    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1))
    nearest_index = [t[0] for t in sorted_dist[:require]]
    addition_set = [t for i, t in enumerate(more.member) if i in set(nearest_index)]
    remain_set = [t for i, t in enumerate(more.member) if i not in set(nearest_index)]
    
    first_ncp, first_mid = NCP_dis_merge(less, addition_set, att_trees, qi_range, is_categorical, qi_len)
    r_middle = middle_group(remain_set, att_trees, is_categorical, qi_len)
    first_ncp += len(remain_set) * NCP(r_middle, att_trees, qi_range, is_categorical, qi_len)
    
    # Second method: merge completely
    second_ncp, second_mid = NCP_dis(less.middle, more.middle, att_trees, qi_range, is_categorical, qi_len)
    second_ncp *= all_length
    
    if first_ncp <= second_ncp:
        less.member.extend(addition_set)
        less.middle = first_mid
        more.member = remain_set
        more.middle = r_middle
        sub_partitions.append(more)
    else:
        less.member.extend(more.member)
        less.middle = second_mid
        less.can_split = False
    sub_partitions.append(less)


def can_split(partition, k):
    """Check if partition can be further split"""
    if partition.can_split is False:
        return False
    if len(partition) < 2 * k:
        return False
    return True


def anonymize(partition, k, att_trees, qi_range, is_categorical, qi_len, result):
    """Main TDG anonymization procedure"""
    if not can_split(partition, k):
        result.append(partition)
        return
    
    u, v = get_pair(partition, att_trees, qi_range, is_categorical, qi_len)
    sub_partitions = distribute_record(u, v, partition, att_trees, qi_range, is_categorical, qi_len)
    
    if len(sub_partitions[0]) < k:
        balance(sub_partitions, 0, k, att_trees, qi_range, is_categorical, qi_len)
    elif len(sub_partitions[1]) < k:
        balance(sub_partitions, 1, k, att_trees, qi_range, is_categorical, qi_len)
    
    for sub_partition in sub_partitions:
        anonymize(sub_partition, k, att_trees, qi_range, is_categorical, qi_len, result)


def tdg_anonymization(data, k, qi_indices, sa_indices, is_categorical):
    """Main TDG k-anonymity algorithm"""
    qi_len = len(qi_indices)
    
    # Build attribute trees
    att_trees = build_hierarchy_trees(data, qi_indices, is_categorical)
    
    # Initialize ranges and middle values
    qi_range = []
    middle = []
    
    for i in range(qi_len):
        if not is_categorical[i]:
            qi_range.append(att_trees[i].range)
            middle.append(att_trees[i].value)
        else:
            qi_range.append(len(att_trees[i]['*']))
            middle.append('*')
    
    # Create initial partition with all data
    whole_partition = Partition(data, middle)
    
    # Perform anonymization
    start_time = time.time()
    result_partitions = []
    anonymize(whole_partition, k, att_trees, qi_range, is_categorical, qi_len, result_partitions)
    runtime = time.time() - start_time
    
    # Generate anonymized dataset
    anonymized_data = []
    ncp = 0.0
    
    for partition in result_partitions:
        # Calculate NCP for this partition
        gen_result = partition.middle
        rncp = NCP(gen_result, att_trees, qi_range, is_categorical, qi_len)
        
        # Add records with generalized values
        for record in partition.member:
            new_record = record[:]
            
            # Replace QI values with generalized values
            for i, qi_idx in enumerate(qi_indices):
                new_record[qi_idx] = gen_result[i]
            
            anonymized_data.append(new_record)
        
        rncp *= len(partition)
        ncp += rncp
    
    # Convert NCP to percentage
    ncp /= len(data)
    ncp /= qi_len
    ncp *= 100
    
    return anonymized_data, ncp, runtime


def load_csv(filename):
    """Load CSV file"""
    data = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def save_csv(filename, data):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description='Standalone Top-Down Greedy (TDG) k-Anonymity Algorithm')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output CSV file for anonymized data')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter (default: 5)')
    parser.add_argument('--qi-columns', required=True, help='Comma-separated quasi-identifier column indices (0-based)')
    parser.add_argument('--sa-columns', default='', help='Comma-separated sensitive attribute column indices (0-based)')
    parser.add_argument('--categorical', default='', help='Comma-separated indices of categorical QI columns')
    parser.add_argument('--header', action='store_true', help='Input file has header row')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
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
    print(f"Random seed: {args.seed}")
    
    # Determine which QI columns are categorical
    is_categorical = []
    for i, qi_idx in enumerate(qi_indices):
        is_categorical.append(qi_idx in categorical_indices)
    
    print(f"Categorical QIs: {[qi_indices[i] for i, cat in enumerate(is_categorical) if cat]}")
    
    # Run TDG anonymization
    print("Running Top-Down Greedy k-anonymity algorithm...")
    anonymized_data, ncp, runtime = tdg_anonymization(data, args.k, qi_indices, sa_indices, is_categorical)
    
    # Add header back if present
    if header:
        anonymized_data.insert(0, header)
    
    # Save results
    save_csv(args.output_file, anonymized_data)
    
    # Print results
    print(f"\nAnonymization completed!")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Information loss (NCP): {ncp:.2f}%")
    print(f"Output saved to: {args.output_file}")
    print(f"Records processed: {len(data)} -> {len(anonymized_data) - (1 if header else 0)}")


if __name__ == "__main__":
    main()