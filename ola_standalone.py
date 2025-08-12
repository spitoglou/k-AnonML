#!/usr/bin/env python3
"""
Standalone Optimal Lattice Anonymization (OLA) k-Anonymity Algorithm

This script implements the OLA algorithm for k-anonymity without external dependencies.
It uses a binary search approach through a generalization lattice to find minimal generalizations.

Usage:
    python ola_standalone.py input.csv output.csv --k 5 --qi-columns 0,1,2 --sa-columns 3

Author: Standalone implementation based on OLA/ELEmam algorithm
"""

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from functools import cmp_to_key


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


def age_generalization(data, level):
    """Age generalization function"""
    return segmentation(data, level, 1, 100, [5, 10, 20, "*"])


def segmentation(data, level, min_num, max_num, div_list):
    """Transforms numerical data to segmented state"""
    ret = []
    if not isinstance(data, list) and not isinstance(data, range):
        values = [int(data)]
    else:
        values = list(map(int, data))

    seg = div_list[level]

    if len(div_list)-1 == level and not isinstance(seg, int):
        return [seg] * len(values)

    groups = range(0, math.floor((max_num+1-min_num)/seg))
    div_max = min_num + seg + seg * groups[-1]

    for value in values:
        if value >= div_max:
            value = div_max - 1

        for i in groups:
            b = min_num + seg * i
            e = b + seg
            if b <= value < e:
                e -= 1
                ret.append(str(b) + "-" + str(e))
                break
    return ret


def substitution_generalization(data, level, wordlists):
    """Hierarchical substitution generalization"""
    ret = []
    if not isinstance(data, list):
        values = [data]
    else:
        values = data

    if level > len(wordlists)-1:
        return ['*'] * len(values)

    wordlist = wordlists[level]

    for value in values:
        for k, v in wordlist.items():
            if value in v:
                ret.append(k)
                break
    return ret


class NumRange:
    """Numeric range for generalization hierarchies"""
    
    def __init__(self, sort_value, support=None):
        self.sort_value = list(sort_value)
        self.support = support or {}
        self.range = float(sort_value[-1]) - float(sort_value[0]) if len(sort_value) > 1 else 0
        self.dict = {v: i for i, v in enumerate(sort_value)}
        self.value = f"{sort_value[0]},{sort_value[-1]}" if len(sort_value) > 1 else str(sort_value[0])


class Node:
    """Node in the generalization lattice"""
    
    def __init__(self, attributes, max_gen_level, node_id=0):
        self.attributes = list(attributes)
        self.level = sum(attributes)
        self.iskanon = None
        self.id = node_id
        self.eqclasses = 0
        self.DM_penalty = 0
        self.DMs_penalty = 0
        
        # Calculate precision
        attribute_count = len(self.attributes)
        self.prec = 0
        for a in range(attribute_count):
            self.prec += (self.attributes[a] / max_gen_level[a]) if max_gen_level[a] > 0 else 0
        self.prec /= attribute_count


def apply_generalization(data, gen_func, level, dictionary, is_categorical=True):
    """Apply generalization strategy to data"""
    tmp_array = []
    gen_index = []
    count = len(dictionary)-1

    for v in data:
        if callable(gen_func):
            vg = gen_func(v, level)
        elif isinstance(gen_func, list) and callable(gen_func[0]):
            args = gen_func[1:]
            vg = gen_func[0](v, level, *args)
        else:
            vg = v

        if isinstance(vg, list):
            vg = vg[0]
            
        if vg not in tmp_array:
            count += 1
            dictionary[count] = vg
            tmp_array.append(vg)
            gen_index.append(count)
        else:
            for key, val in dictionary.items():
                if vg == val:
                    gen_index.append(key)
                    break

    return gen_index


def build_hierarchy_trees(data, qi_indices, gen_strategies, max_gen_levels, is_categorical):
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
            # Categorical attribute - build hierarchy from generalization strategy
            unique_values = list(set(column_values))
            dictionary = {k: v for k, v in enumerate(unique_values)}
            dict_reverse = {v: k for k, v in dictionary.items()}
            
            # Build hierarchy levels
            hierarchy = [list(dictionary.keys())]  # Level 0
            
            quasi_identifier_set = set(column_values)
            for level in range(max_gen_levels[i]):
                gen_indices = apply_generalization(
                    quasi_identifier_set, gen_strategies[i], level, dictionary)
                hierarchy.append(gen_indices)
            
            att_trees.append(hierarchy)
    
    return att_trees


class AnonCheck:
    """Anonymity checker for OLA algorithm"""
    
    def __init__(self, raw_data, max_gen, gen_strat, allowed_suppressed, k, qi_indices):
        self.k = k
        self.allowed_suppressed = allowed_suppressed
        self.raw_rows_count = len(raw_data)
        self.raw_cols_count = len(raw_data[0]) if raw_data else 0
        self.qi_indices = qi_indices
        
        self.hier_array = []
        self.data_array = []
        dict_array = []

        self.buffer = [[0 for _ in range(len(qi_indices))] for _ in range(self.raw_rows_count)]
        self.prev_gen_to_apply = None
        self.eq_classes_dict = None
        self.is_transformed = True

        key_rows = range(self.raw_rows_count)
        vals = [1] * self.raw_rows_count
        self.dummy_raw_eq_classes = list(zip(vals, key_rows))

        # Convert data to indexed format for quasi-identifiers only
        self.test = []
        for r in range(self.raw_rows_count):
            self.test.append([raw_data[r][qi_idx] for qi_idx in qi_indices])

        for col in range(len(qi_indices)):
            qi_idx = qi_indices[col]
            quasi_identifier = [raw_data[r][qi_idx] for r in range(self.raw_rows_count)]
            quasi_identifier_set = set(quasi_identifier)

            dict_array.append({k: v for k, v in enumerate(quasi_identifier_set)})
            dict_reverse = {v: k for k, v in dict_array[col].items()}
            
            data_col = [dict_reverse[v] for v in quasi_identifier]
            self.data_array.append(data_col)

            # Build hierarchy levels
            self.hier_array.append([list(dict_array[-1].keys())])
            for level in range(max_gen[col]):
                gen_indices = apply_generalization(
                    quasi_identifier_set, gen_strat[col], level, dict_array[-1])
                self.hier_array[col].append(gen_indices)

    def calculate_kanon(self, node):
        """Check if node satisfies k-anonymity"""
        gen_to_apply = node.attributes
        is_rollup_allowed = False

        raw_eq_classes = self.dummy_raw_eq_classes

        if self.eq_classes_dict is not None:
            prev_level = sum(self.prev_gen_to_apply)
            level = sum(gen_to_apply)
            if level > prev_level:
                is_rollup_allowed = True
                for num in range(len(gen_to_apply)):
                    if gen_to_apply[num] < self.prev_gen_to_apply[num]:
                        is_rollup_allowed = False
                        break

                if is_rollup_allowed:
                    self.is_transformed = False
                    raw_eq_classes = self.eq_classes_dict.values()

        # Projection
        cols_to_iterate = range(len(self.qi_indices))
        if self.is_transformed or is_rollup_allowed:
            if self.prev_gen_to_apply is not None:
                cols_to_iterate = [i for i, value in enumerate(gen_to_apply) 
                                 if value != self.prev_gen_to_apply[i]]

        if not is_rollup_allowed:
            self.is_transformed = True

        self.eq_classes_dict = {}
        update = self.eq_classes_dict.update

        for val, key_row in raw_eq_classes:
            tmp = self.buffer[key_row]
            rawr = self.test[key_row]
            for col in cols_to_iterate:
                tmp[col] = self.hier_array[col][gen_to_apply[col]][rawr[col]]

            tup = tuple(tmp)

            try:
                self.eq_classes_dict[tup][0] += val
            except KeyError:
                update({tup: [val, key_row]})

        self.prev_gen_to_apply = gen_to_apply.copy()

        suppressed_count = 0
        eq_classes = sorted(list(zip(*self.eq_classes_dict.values()))[0])
        eqsum = 0
        amount = 0
        
        for v in eq_classes:
            if v < self.k:
                suppressed_count += v
                if not suppressed_count > self.allowed_suppressed:
                    node.DM_penalty += v * self.raw_rows_count
            else:
                eqsum += v
                amount += 1
                node.DM_penalty += v*v
                node.DMs_penalty += v*v
                
        if amount == 0:
            node.eqclasses = 0
        else:
            node.eqclasses = eqsum/amount
            
        suppressed_count = 0
        for v in eq_classes:
            if v < self.k:
                if self.allowed_suppressed == 0:
                    return False
                suppressed_count += v
                if suppressed_count > self.allowed_suppressed:
                    return False
            else:
                return True
        return True


def create_nodes(qi_len, max_gen_levels):
    """Create all possible nodes in generalization lattice"""
    nodes = []
    node_id = 0
    
    def generate_combinations(current_attrs, remaining_dims):
        nonlocal node_id
        if remaining_dims == 0:
            node = Node(current_attrs, max_gen_levels, node_id)
            nodes.append(node)
            node_id += 1
            return
        
        dim_idx = qi_len - remaining_dims
        for level in range(max_gen_levels[dim_idx] + 1):
            new_attrs = current_attrs + [level]
            generate_combinations(new_attrs, remaining_dims - 1)
    
    generate_combinations([], qi_len)
    return nodes


def sort_nodes(nodes):
    """Sort nodes by level in generalization lattice"""
    sorted_array = []
    part_array = {}
    
    for n in reversed(nodes):
        h = n.level
        if h not in part_array:
            part_array[h] = []
        part_array[h].append(n)

    for level in sorted(part_array.keys()):
        sorted_array.extend(part_array[level])

    return sorted_array


def evaluate_ola(nodes, ac, min_k_nodes):
    """Main OLA evaluation algorithm using binary search"""
    if not nodes:
        return True
        
    # Binary search approach
    low = 0
    high = len(nodes) - 1
    
    while low <= high:
        mid = (low + high) // 2
        node = nodes[mid]
        
        if node.iskanon is None:
            if ac.calculate_kanon(node):
                node.iskanon = True
                # Check if this is a minimal node
                is_minimal = True
                for existing_node in min_k_nodes[:]:
                    # Check dominance relationships
                    dominates = all(node.attributes[i] <= existing_node.attributes[i] 
                                  for i in range(len(node.attributes)))
                    dominated_by = all(existing_node.attributes[i] <= node.attributes[i] 
                                     for i in range(len(node.attributes)))
                    
                    if dominates and node.level < existing_node.level:
                        min_k_nodes.remove(existing_node)
                    elif dominated_by and existing_node.level < node.level:
                        is_minimal = False
                        break
                
                if is_minimal:
                    min_k_nodes.append(node)
                
                # Continue search in lower half
                high = mid - 1
            else:
                node.iskanon = False
                # Continue search in upper half  
                low = mid + 1
        else:
            # Node already evaluated, adjust search accordingly
            if node.iskanon:
                high = mid - 1
            else:
                low = mid + 1
    
    return True


def ola_anonymization(data, k, qi_indices, sa_indices, gen_strategies, max_gen_levels, 
                     is_categorical, suppression_rate=0):
    """Main OLA k-anonymity algorithm"""
    qi_len = len(qi_indices)
    allowed_suppressed = int(len(data) * (suppression_rate / 100))
    
    # Create anonymity checker
    ac = AnonCheck(data, max_gen_levels, gen_strategies, allowed_suppressed, k, qi_indices)
    
    # Create all nodes in generalization lattice
    nodes = create_nodes(qi_len, max_gen_levels)
    sorted_nodes = sort_nodes(nodes)
    
    # Find minimal k-anonymous generalizations
    start_time = time.time()
    min_k_nodes = []
    evaluate_ola(sorted_nodes, ac, min_k_nodes)
    runtime = time.time() - start_time
    
    if not min_k_nodes:
        # Fallback: use most general node
        max_attrs = max_gen_levels[:]
        general_node = Node(max_attrs, max_gen_levels)
        min_k_nodes = [general_node]
    
    # Choose best node based on precision
    best_node = min(min_k_nodes, key=lambda n: n.prec)
    
    # Generate anonymized dataset
    anonymized_data = []
    for row_idx, original_row in enumerate(data):
        new_row = original_row[:]
        
        # Apply generalizations to QI columns
        for i, qi_idx in enumerate(qi_indices):
            level = best_node.attributes[i]
            original_value = original_row[qi_idx]
            
            if level == 0:
                generalized_value = original_value
            else:
                if callable(gen_strategies[i]):
                    gen_result = gen_strategies[i](original_value, level - 1)
                elif isinstance(gen_strategies[i], list) and callable(gen_strategies[i][0]):
                    args = gen_strategies[i][1:]
                    gen_result = gen_strategies[i][0](original_value, level - 1, *args)
                else:
                    gen_result = original_value
                
                if isinstance(gen_result, list):
                    generalized_value = gen_result[0]
                else:
                    generalized_value = gen_result
            
            new_row[qi_idx] = generalized_value
        
        anonymized_data.append(new_row)
    
    # Calculate information loss (simplified NCP)
    ncp = sum(best_node.attributes) / (sum(max_gen_levels) * qi_len) * 100 if sum(max_gen_levels) > 0 else 0
    
    return anonymized_data, ncp, runtime


def create_default_generalization_strategies(data, qi_indices, is_categorical):
    """Create default generalization strategies based on data types"""
    strategies = []
    max_levels = []
    
    for i, qi_idx in enumerate(qi_indices):
        column_values = [row[qi_idx] for row in data]
        unique_values = list(set(column_values))
        
        if not is_categorical[i]:
            # Try to detect if it's age-like data
            try:
                numeric_values = [float(v) for v in unique_values]
                min_val, max_val = min(numeric_values), max(numeric_values)
                
                # Simple numeric generalization
                def numeric_gen(data, level, min_v=min_val, max_v=max_val):
                    try:
                        val = float(data)
                        if level == 0:
                            ranges = [(min_v + i*10, min_v + (i+1)*10) for i in range(int((max_v - min_v)/10) + 1)]
                        elif level == 1:
                            ranges = [(min_v + i*20, min_v + (i+1)*20) for i in range(int((max_v - min_v)/20) + 1)]
                        else:
                            return ["*"]
                        
                        for start, end in ranges:
                            if start <= val < end:
                                return [f"{int(start)}-{int(end-1)}"]
                        return [f"{int(min_v)}-{int(max_v)}"]
                    except:
                        return [str(data)]
                
                strategies.append(numeric_gen)
                max_levels.append(3)
                
            except ValueError:
                # Non-numeric data, treat as categorical
                strategies.append(lambda data, level: ["*"] if level > 0 else [str(data)])
                max_levels.append(1)
        else:
            # Simple categorical generalization
            strategies.append(lambda data, level: ["*"] if level > 0 else [str(data)])
            max_levels.append(1)
    
    return strategies, max_levels


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
    parser = argparse.ArgumentParser(description='Standalone OLA (Optimal Lattice Anonymization) k-Anonymity Algorithm')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output CSV file for anonymized data')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter (default: 5)')
    parser.add_argument('--qi-columns', required=True, help='Comma-separated quasi-identifier column indices (0-based)')
    parser.add_argument('--sa-columns', default='', help='Comma-separated sensitive attribute column indices (0-based)')
    parser.add_argument('--categorical', default='', help='Comma-separated indices of categorical QI columns')
    parser.add_argument('--header', action='store_true', help='Input file has header row')
    parser.add_argument('--suppression-rate', type=float, default=0, help='Maximum suppression rate percentage (default: 0)')
    
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
    print(f"Suppression rate: {args.suppression_rate}%")
    
    # Determine which QI columns are categorical
    is_categorical = []
    for i, qi_idx in enumerate(qi_indices):
        is_categorical.append(qi_idx in categorical_indices)
    
    print(f"Categorical QIs: {[qi_indices[i] for i, cat in enumerate(is_categorical) if cat]}")
    
    # Create generalization strategies
    gen_strategies, max_gen_levels = create_default_generalization_strategies(
        data, qi_indices, is_categorical)
    
    print(f"Max generalization levels: {max_gen_levels}")
    
    # Run OLA anonymization
    print("Running OLA (Optimal Lattice Anonymization) k-anonymity algorithm...")
    anonymized_data, ncp, runtime = ola_anonymization(
        data, args.k, qi_indices, sa_indices, gen_strategies, 
        max_gen_levels, is_categorical, args.suppression_rate)
    
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