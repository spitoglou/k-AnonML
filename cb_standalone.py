#!/usr/bin/env python3
"""
Standalone Clustering-Based (CB) k-Anonymity Algorithm

This script implements the clustering-based k-anonymity algorithm without external dependencies.
It uses k-NN clustering to group similar records and then generalizes each cluster.

Usage:
    python cb_standalone.py input.csv output.csv --k 5 --qi-columns 0,1,2 --sa-columns 3

Author: Standalone implementation based on Clustering-Based k-anonymity algorithm
"""

import argparse
import csv
import operator
import random
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


def get_num_list_from_str(stemp):
    """Convert string to numeric list"""
    try:
        float(stemp)
        return [stemp]
    except ValueError:
        return stemp.split(',')


class NumRange:
    """Numeric range for generalization hierarchies"""
    
    def __init__(self, sort_value, support=None):
        self.sort_value = list(sort_value)
        self.support = support or {}
        self.range = float(sort_value[-1]) - float(sort_value[0]) if len(sort_value) > 1 else 0
        self.dict = {v: i for i, v in enumerate(sort_value)}


class Cluster:
    """Cluster for clustering-based k-anonymity"""
    
    def __init__(self, member, gen_result, information_loss=0.0):
        self.information_loss = information_loss
        self.member = list(member)
        self.gen_result = gen_result[:]
        self.center = gen_result[:]
        
        # Calculate center for numeric attributes
        for i in range(len(gen_result)):
            if not IS_CAT[i]:
                try:
                    self.center[i] = str(sum([float(t[i]) for t in self.member]) / len(self.member))
                except (ValueError, ZeroDivisionError):
                    self.center[i] = gen_result[i]

    def add_record(self, record):
        """Add record to cluster"""
        self.member.append(record)
        self.update_gen_result(record, record)

    def update_cluster(self):
        """Update cluster information when member is changed"""
        self.gen_result = cluster_generalization(self.member)
        for i in range(len(self.gen_result)):
            if IS_CAT[i]:
                self.center[i] = self.gen_result[i]
            else:
                try:
                    self.center[i] = str(sum([float(t[i]) for t in self.member]) / len(self.member))
                except (ValueError, ZeroDivisionError):
                    self.center[i] = self.gen_result[i]
        self.information_loss = len(self.member) * NCP(self.gen_result)

    def update_gen_result(self, merge_gen_result, center, num=1):
        """Update gen_result and information_loss after adding record or merging cluster"""
        self.gen_result = generalization(self.gen_result, merge_gen_result)
        current_len = len(self.member)
        for i in range(len(self.gen_result)):
            if IS_CAT[i]:
                self.center[i] = self.gen_result[i]
            else:
                try:
                    old_center = float(self.center[i])
                    new_center = float(center[i])
                    self.center[i] = str((old_center * (current_len - num) + new_center * num) / current_len)
                except (ValueError, ZeroDivisionError):
                    self.center[i] = self.gen_result[i]
        self.information_loss = len(self.member) * NCP(self.gen_result)

    def merge_cluster(self, cluster):
        """Merge cluster into self"""
        self.member.extend(cluster.member)
        self.update_gen_result(cluster.gen_result, cluster.center, len(cluster))

    def __len__(self):
        """Return number of records in cluster"""
        return len(self.member)

    def __str__(self):
        return str(self.gen_result)


def qid_to_key(record):
    """Convert QI record to string key for caching"""
    return ','.join(str(x) for x in record)


def r_distance(source, target):
    """Return distance between source and target based on NCP"""
    source_gen = source
    target_gen = target
    source_len = 1
    target_len = 1
    
    # Check if target is Cluster
    if isinstance(target, Cluster):
        target_gen = target.gen_result
        target_len = len(target)
    # Check if source is Cluster
    if isinstance(source, Cluster):
        source_gen = source.gen_result
        source_len = len(source)
        
    if source_gen == target_gen:
        return 0
        
    gen = generalization(source_gen, target_gen)
    distance = (source_len + target_len) * NCP(gen)
    return distance


def diff_distance(record, cluster):
    """Return IL(cluster and record) - IL(cluster)"""
    gen_after = generalization(record, cluster.gen_result)
    return NCP(gen_after) * (len(cluster) + 1) - cluster.information_loss


def NCP(record):
    """Compute NCP (Normalized Certainty Penalty)"""
    if not record or not QI_RANGE:
        return 0.0
        
    ncp = 0.0
    list_key = qid_to_key(record)
    
    if list_key in NCP_CACHE:
        return NCP_CACHE[list_key]
    
    for i in range(min(len(record), len(QI_RANGE))):
        width = 0.0
        if not IS_CAT[i]:
            try:
                float(record[i])
                width = 0.0  # Single value has no width
            except ValueError:
                temp = record[i].split(',')
                if len(temp) >= 2:
                    try:
                        width = float(temp[1]) - float(temp[0])
                    except ValueError:
                        width = 0.0
        else:
            # For categorical, estimate width
            if record[i] == '*':
                width = QI_RANGE[i]
            else:
                width = 0.0
        
        if QI_RANGE[i] > 0:
            width /= QI_RANGE[i]
        ncp += width
    
    NCP_CACHE[list_key] = ncp
    return ncp


def get_LCA(index, item1, item2):
    """Get lowest common ancestor for categorical attributes"""
    if item1 == item2:
        return item1
    
    cache_key = item1 + item2
    if index < len(LCA_CACHE) and cache_key in LCA_CACHE[index]:
        return LCA_CACHE[index][cache_key]
    
    # Simple LCA for standalone implementation
    result = '*'  # Most general value
    
    if index < len(LCA_CACHE):
        LCA_CACHE[index][cache_key] = result
    
    return result


def generalization(record1, record2):
    """Compute generalization result of record1 and record2"""
    gen = []
    qi_len = min(len(record1), len(record2))
    
    for i in range(qi_len):
        if not IS_CAT[i]:
            # Numeric generalization
            split_number = []
            split_number.extend(get_num_list_from_str(record1[i]))
            split_number.extend(get_num_list_from_str(record2[i]))
            split_number = list(set(split_number))
            
            if len(split_number) == 1:
                gen.append(split_number[0])
            else:
                split_number.sort(key=cmp_to_key(cmp_str))
                gen.append(split_number[0] + ',' + split_number[-1])
        else:
            # Categorical generalization
            gen.append(get_LCA(i, record1[i], record2[i]))
    
    return gen


def cluster_generalization(records):
    """Calculate generalization result of records list recursively"""
    if not records:
        return []
    
    gen = records[0][:]
    for i in range(1, len(records)):
        gen = generalization(gen, records[i])
    return gen


def find_best_knn(index, k, data, max_candidates=100):
    """Find k nearest neighbors of record, remove them from data (optimized)"""
    if not data or index >= len(data):
        return None, []
    
    record = data[index]
    data_len = len(data)
    
    # For large datasets, sample candidates to avoid O(n²) complexity
    if data_len > max_candidates:
        # Sample random candidates plus some nearby indices
        sample_size = min(max_candidates, data_len - 1)
        candidate_indices = set()
        
        # Add random sample
        while len(candidate_indices) < sample_size // 2:
            idx = random.randrange(data_len)
            if idx != index:
                candidate_indices.add(idx)
        
        # Add nearby indices (locality assumption)
        for offset in range(1, sample_size // 2 + 1):
            if len(candidate_indices) >= sample_size:
                break
            for direction in [1, -1]:
                new_idx = (index + direction * offset) % data_len
                if new_idx != index:
                    candidate_indices.add(new_idx)
                    if len(candidate_indices) >= sample_size:
                        break
        
        candidates = list(candidate_indices)
    else:
        candidates = [i for i in range(data_len) if i != index]
    
    # Calculate distances only for candidates
    dist_dict = {}
    for i in candidates:
        dist = r_distance(record, data[i])
        dist_dict[i] = dist
    
    # Sort by distance and select k-1 nearest neighbors
    sorted_dict = sorted(dist_dict.items(), key=operator.itemgetter(1))
    knn = sorted_dict[:k - 1]
    knn.append((index, 0))  # Add the seed record itself
    
    record_index = [t[0] for t in knn]
    elements = [data[t[0]] for t in knn]
    
    gen = cluster_generalization(elements)
    cluster = Cluster(elements, gen, k * NCP(gen))
    
    return cluster, record_index


def find_best_cluster_iloss(record, clusters):
    """Find best cluster for record (residual assignment)"""
    if not clusters:
        return 0
    
    min_distance = float('inf')
    min_index = 0
    
    for i, cluster in enumerate(clusters):
        distance = r_distance(record, cluster.gen_result)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    
    return min_index


def find_best_cluster_iloss_increase(record, clusters):
    """Find best cluster for record based on information loss increase"""
    if not clusters:
        return 0
    
    min_diff = float('inf')
    min_index = 0
    
    for i, cluster in enumerate(clusters):
        diff = diff_distance(record, cluster)
        if diff < min_diff:
            min_diff = diff
            min_index = i
    
    return min_index


def find_furthest_record(record, data):
    """Find the furthest record from the given record in data"""
    max_distance = 0
    max_index = -1
    for index in range(len(data)):
        current_distance = r_distance(record, data[index])
        if current_distance >= max_distance:
            max_distance = current_distance
            max_index = index
    return max_index


def find_best_record_iloss_increase(cluster, data):
    """Find record that causes minimum information loss increase when added to cluster"""
    min_diff = float('inf')
    min_index = 0
    for index, record in enumerate(data):
        diff = diff_distance(record, cluster)
        if diff < min_diff:
            min_diff = diff
            min_index = index
    return min_index


def clustering_knn(data, k=25):
    """Group records using k-NN clustering (optimized)"""
    clusters = []
    data_copy = data[:]
    max_iterations = len(data_copy) // k + 10  # Prevent infinite loops
    iterations = 0
    
    # Create clusters of size k
    while len(data_copy) >= k and iterations < max_iterations:
        index = random.randrange(len(data_copy))
        cluster, record_index = find_best_knn(index, k, data_copy)
        
        if cluster is None or not record_index:
            break
        
        # Remove used records from data (sort in reverse to avoid index issues)
        record_index.sort(reverse=True)
        for idx in record_index:
            if idx < len(data_copy):
                data_copy.pop(idx)
        
        clusters.append(cluster)
        iterations += 1
    
    # Residual assignment - assign remaining records to existing clusters
    while data_copy and clusters:
        record = data_copy.pop()
        cluster_index = find_best_cluster_iloss(record, clusters)
        clusters[cluster_index].add_record(record)
    
    # If no clusters exist and we still have data, create one cluster with all remaining records
    if not clusters and data_copy:
        # Create single cluster with all remaining records
        gen = cluster_generalization(data_copy)
        cluster = Cluster(data_copy, gen, len(data_copy) * NCP(gen))
        clusters.append(cluster)
    
    return clusters


def clustering_kmember(data, k=25):
    """Group records using k-member clustering algorithm"""
    clusters = []
    data_copy = data[:]
    
    # Randomly choose initial seed
    r_pos = random.randrange(len(data_copy))
    r_i = data_copy[r_pos]
    
    while len(data_copy) >= k:
        # Find furthest record from current seed
        r_pos = find_furthest_record(r_i, data_copy)
        r_i = data_copy.pop(r_pos)
        
        # Create cluster with this seed
        cluster = Cluster([r_i], r_i[:], NCP(r_i))
        
        # Add k-1 more records that minimize information loss increase
        while len(cluster) < k and data_copy:
            r_pos = find_best_record_iloss_increase(cluster, data_copy)
            r_j = data_copy.pop(r_pos)
            cluster.add_record(r_j)
        
        clusters.append(cluster)
    
    # Residual assignment - assign remaining records to existing clusters
    while data_copy and clusters:
        record = data_copy.pop()
        cluster_index = find_best_cluster_iloss_increase(record, clusters)
        clusters[cluster_index].add_record(record)
    
    # If no clusters exist and we still have data, create one cluster
    if not clusters and data_copy:
        gen = cluster_generalization(data_copy)
        cluster = Cluster(data_copy, gen, len(data_copy) * NCP(gen))
        clusters.append(cluster)
    
    return clusters


def adjust_cluster(cluster, residual, k):
    """Adjust cluster size by moving records to residual if cluster is too large"""
    if len(cluster) <= k:
        return
    
    center = cluster.center
    dist_dict = {}
    
    # Calculate distances from center for all records
    for i, record in enumerate(cluster.member):
        dist = r_distance(center, record)
        dist_dict[i] = dist
    
    # Sort by distance and keep only k closest records
    sorted_dist = sorted(dist_dict.items(), key=operator.itemgetter(1))
    keep_indices = [t[0] for t in sorted_dist[:k]]
    
    # Move distant records to residual
    new_members = []
    for i, record in enumerate(cluster.member):
        if i in keep_indices:
            new_members.append(record)
        else:
            residual.append(record)
    
    # Update cluster
    cluster.member = new_members
    cluster.update_cluster()


def clustering_oka(data, k=25):
    """Group records using OKA (One-time pass K-means) algorithm"""
    clusters = []
    can_clusters = []
    less_clusters = []
    data_copy = data[:]
    
    # Randomly choose seeds for initial clusters
    if len(data_copy) < k:
        # Not enough data for clustering
        gen = cluster_generalization(data_copy)
        cluster = Cluster(data_copy, gen, len(data_copy) * NCP(gen))
        return [cluster]
    
    num_seeds = max(1, len(data_copy) // k)
    seed_indices = random.sample(range(len(data_copy)), min(num_seeds, len(data_copy)))
    
    # Create initial clusters with seeds
    for index in seed_indices:
        record = data_copy[index]
        can_clusters.append(Cluster([record], record[:], NCP(record)))
    
    # Remove seeds from data
    data_copy = [t for i, t in enumerate(data_copy) if i not in set(seed_indices)]
    
    # Assign remaining records to clusters
    while data_copy:
        record = data_copy.pop()
        index = find_best_cluster_iloss(record, can_clusters)
        can_clusters[index].add_record(record)
    
    # Process clusters based on size
    residual = []
    for cluster in can_clusters:
        if len(cluster) < k:
            less_clusters.append(cluster)
        else:
            if len(cluster) > k:
                adjust_cluster(cluster, residual, k)
            clusters.append(cluster)
    
    # Handle residual records
    while residual:
        record = residual.pop()
        if less_clusters:
            # Add to under-sized cluster
            index = find_best_cluster_iloss(record, less_clusters)
            less_clusters[index].add_record(record)
            # Move to final clusters if now has enough members
            if len(less_clusters[index]) >= k:
                clusters.append(less_clusters.pop(index))
        else:
            # Add to existing clusters
            if clusters:
                index = find_best_cluster_iloss(record, clusters)
                clusters[index].add_record(record)
    
    # Move any remaining under-sized clusters to final list
    clusters.extend(less_clusters)
    
    return clusters


def build_hierarchy_trees(data, qi_indices, is_categorical):
    """Build attribute hierarchy trees from data"""
    att_trees = []
    global QI_RANGE, IS_CAT, LCA_CACHE
    
    QI_RANGE = []
    IS_CAT = is_categorical[:]
    LCA_CACHE = [dict() for _ in range(len(qi_indices))]
    
    for i, qi_idx in enumerate(qi_indices):
        column_values = [row[qi_idx] for row in data]
        
        if not is_categorical[i]:
            # Numeric attribute
            unique_values = list(set(column_values))
            try:
                unique_values.sort(key=float)
                att_range = float(unique_values[-1]) - float(unique_values[0]) if len(unique_values) > 1 else 0
            except ValueError:
                unique_values.sort()
                att_range = len(unique_values) - 1
            
            support = defaultdict(int)
            for val in column_values:
                support[val] += 1
            
            att_trees.append(NumRange(unique_values, dict(support)))
            QI_RANGE.append(att_range)
        else:
            # Categorical attribute
            unique_values = list(set(column_values))
            hierarchy = {'*': unique_values}
            for val in unique_values:
                hierarchy[val] = [val]
            att_trees.append(hierarchy)
            QI_RANGE.append(len(unique_values))
    
    return att_trees


def cb_anonymization(data, k, qi_indices, sa_indices, is_categorical, algorithm_type='knn'):
    """Main clustering-based k-anonymity algorithm"""
    global NCP_CACHE, QI_LEN
    
    QI_LEN = len(qi_indices)
    NCP_CACHE = {}
    
    # Extract QI data
    qi_data = []
    for row in data:
        qi_row = [row[qi_idx] for qi_idx in qi_indices]
        qi_data.append(qi_row)
    
    # Build attribute trees
    att_trees = build_hierarchy_trees(data, qi_indices, is_categorical)
    
    # Perform clustering
    start_time = time.time()
    
    if algorithm_type == 'knn':
        clusters = clustering_knn(qi_data, k)
    elif algorithm_type == 'kmember':
        clusters = clustering_kmember(qi_data, k)
    elif algorithm_type == 'oka':
        clusters = clustering_oka(qi_data, k)
    else:
        print(f"Unknown algorithm type: {algorithm_type}, using k-NN")
        clusters = clustering_knn(qi_data, k)
    
    runtime = time.time() - start_time
    
    # Generate anonymized dataset
    anonymized_data = []
    ncp = 0.0
    
    # Create mapping from QI values to original records for efficient lookup
    qi_to_original = {}
    for row_idx, row in enumerate(data):
        qi_tuple = tuple(row[qi_idx] for qi_idx in qi_indices)
        if qi_tuple not in qi_to_original:
            qi_to_original[qi_tuple] = []
        qi_to_original[qi_tuple].append((row_idx, row))
    
    for cluster in clusters:
        # For each record in cluster, create anonymized version
        for member_qi in cluster.member:
            qi_tuple = tuple(member_qi)
            
            if qi_tuple in qi_to_original and qi_to_original[qi_tuple]:
                # Get the first available original record with these QI values
                row_idx, original_row = qi_to_original[qi_tuple].pop(0)
                new_row = original_row[:]
                
                # Replace QI values with generalized values
                for j, qi_idx in enumerate(qi_indices):
                    new_row[qi_idx] = cluster.gen_result[j]
                
                anonymized_data.append(new_row)
        
        ncp += cluster.information_loss
    
    # Convert NCP to percentage
    if len(data) > 0 and QI_LEN > 0:
        ncp = (ncp / len(data)) / QI_LEN * 100
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


# Global variables for the algorithm
ATT_TREES = []
QI_RANGE = []
IS_CAT = []
QI_LEN = 0
LCA_CACHE = []
NCP_CACHE = {}


def main():
    parser = argparse.ArgumentParser(description='Standalone Clustering-Based (CB) k-Anonymity Algorithm')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output CSV file for anonymized data')
    parser.add_argument('--k', type=int, default=5, help='k-anonymity parameter (default: 5)')
    parser.add_argument('--qi-columns', required=True, help='Comma-separated quasi-identifier column indices (0-based)')
    parser.add_argument('--sa-columns', default='', help='Comma-separated sensitive attribute column indices (0-based)')
    parser.add_argument('--categorical', default='', help='Comma-separated indices of categorical QI columns')
    parser.add_argument('--header', action='store_true', help='Input file has header row')
    parser.add_argument('--cb-alg', choices=['knn', 'kmember', 'oka'], default='knn',
                       help='Algorithm for cluster based anonymization: knn (k-nearest neighbor), kmember (k-member), oka (one-time k-means) (default: knn)')
    # Alias for backward compatibility
    parser.add_argument('--algorithm', choices=['knn', 'kmember', 'oka'], dest='cb_alg', help=argparse.SUPPRESS)
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
    print(f"Algorithm: {args.cb_alg}")
    print(f"Random seed: {args.seed}")
    
    # Determine which QI columns are categorical
    is_categorical = []
    for i, qi_idx in enumerate(qi_indices):
        is_categorical.append(qi_idx in categorical_indices)
    
    print(f"Categorical QIs: {[qi_indices[i] for i, cat in enumerate(is_categorical) if cat]}")
    
    # Run CB anonymization
    print(f"Running Clustering-Based ({args.cb_alg}) k-anonymity algorithm...")
    anonymized_data, ncp, runtime = cb_anonymization(
        data, args.k, qi_indices, sa_indices, is_categorical, args.cb_alg)
    
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