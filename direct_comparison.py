#!/usr/bin/env python3
"""
Direct Algorithm Comparison: Original vs Standalone Implementations

This script directly calls the original algorithm functions and compares them
with standalone implementations on the 1000-row adult subset dataset.
"""

import csv
import json
import os
import sys
import time
import copy
from typing import Dict, List, Tuple, Any

# Add project paths for imports
sys.path.insert(0, os.getcwd())

# Import original algorithm functions
from basic_mondrian.anonymizer import get_result_one as mondrian_get_result_one
from top_down_greedy.anonymizer import tdg_get_result_one
from clustering_based.anonymizer import get_result_one as cb_get_result_one

# Import data utilities
from utils.data import read_raw
from basic_mondrian.utils.read_adult_data import read_tree


class DirectAlgorithmComparison:
    """Direct comparison between original and standalone implementations"""
    
    def __init__(self):
        self.test_file = "adult_subset_1000.csv"
        self.k = 5
        self.qi_indices = [0, 1, 3, 4, 5, 6, 7]  # ID, sex, race, marital-status, education, native-country, workclass
        self.sa_indices = [9]  # salary-class as sensitive attribute  
        self.categorical_indices = {1, 3, 4, 5, 6, 7}  # sex, race, marital-status, education, native-country, workclass
        self.results = {}
        
    def load_test_data(self):
        """Load the 1000-row test dataset"""
        data = []
        with open(self.test_file, 'r', encoding='utf-8') as f:
            # Detect delimiter
            first_line = f.readline()
            f.seek(0)
            delimiter = ';' if ';' in first_line else ','
            
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)  # Skip header
            for row in reader:
                data.append(row)
        return data
    
    def build_attribute_trees(self, data):
        """Build attribute trees for the test data (simplified version of read_tree)"""
        # For this comparison, we'll use simplified trees
        # In production, you'd use the full generalization hierarchies
        att_trees = []
        
        for i, qi_idx in enumerate(self.qi_indices):
            if qi_idx not in self.categorical_indices:
                # Numeric attribute - create simple range
                column_values = [row[qi_idx] for row in data]
                unique_values = list(set(column_values))
                try:
                    unique_values.sort(key=float)
                except ValueError:
                    unique_values.sort()
                
                # Create simple NumRange-like structure
                att_tree = {
                    'sort_value': unique_values,
                    'range': len(unique_values) - 1 if len(unique_values) > 1 else 0
                }
                att_trees.append(att_tree)
            else:
                # Categorical attribute - create simple hierarchy
                column_values = [row[qi_idx] for row in data]
                unique_values = list(set(column_values))
                
                # Create simple hierarchy
                hierarchy = {'*': unique_values}
                for val in unique_values:
                    hierarchy[val] = [val]
                att_trees.append(hierarchy)
        
        return att_trees
    
    def run_original_mondrian(self, data, att_trees):
        """Run original Mondrian implementation"""
        print("  Running original Mondrian...")
        try:
            start_time = time.time()
            result = mondrian_get_result_one(
                att_trees=att_trees,
                data=data,
                k=self.k,
                path="temp",
                qi_index=self.qi_indices,
                SA_index=self.sa_indices
            )
            runtime = time.time() - start_time
            
            # Calculate NCP from result if possible
            ncp = self._estimate_ncp_from_result(result, data)
            
            return {
                'status': 'success',
                'runtime': runtime,
                'ncp': ncp,
                'records': len(result) if result else 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'runtime': 0.0
            }
    
    def run_original_tdg(self, data, att_trees):
        """Run original TDG implementation"""
        print("  Running original TDG...")
        try:
            start_time = time.time()
            result = tdg_get_result_one(
                att_trees=att_trees,
                data=data,
                k=self.k,
                path="temp",
                qi_index=self.qi_indices,
                SA_index=self.sa_indices
            )
            runtime = time.time() - start_time
            
            # Calculate NCP from result if possible
            ncp = self._estimate_ncp_from_result(result, data)
            
            return {
                'status': 'success',
                'runtime': runtime,
                'ncp': ncp,
                'records': len(result) if result else 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'runtime': 0.0
            }
    
    def run_original_cb(self, data, att_trees, algorithm_type):
        """Run original CB implementation"""
        print(f"  Running original CB ({algorithm_type})...")
        try:
            start_time = time.time()
            result = cb_get_result_one(
                att_trees=att_trees,
                data=data,
                k=self.k,
                path="temp",
                qi_index=self.qi_indices,
                SA_index=self.sa_indices,
                type_alg=algorithm_type
            )
            runtime = time.time() - start_time
            
            # Calculate NCP from result if possible
            ncp = self._estimate_ncp_from_result(result, data)
            
            return {
                'status': 'success',
                'runtime': runtime,
                'ncp': ncp,
                'records': len(result) if result else 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'runtime': 0.0
            }
    
    def run_standalone_algorithm(self, algorithm, script_name, extra_args=None):
        """Run standalone algorithm"""
        print(f"  Running standalone {algorithm}...")
        if extra_args is None:
            extra_args = []
            
        try:
            import subprocess
            
            output_file = f"temp_standalone_{algorithm}_output.csv"
            cmd = [
                sys.executable, script_name,
                self.test_file, output_file,
                "--k", str(self.k),
                "--qi-columns", ",".join(map(str, self.qi_indices)),
                "--categorical", ",".join(map(str, self.categorical_indices)),
                "--header"
            ] + extra_args
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            runtime = time.time() - start_time
            
            if result.returncode == 0:
                ncp = self._extract_ncp_from_output(result.stdout)
                record_count = self._count_records(output_file)
                
                return {
                    'status': 'success',
                    'runtime': runtime,
                    'ncp': ncp,
                    'records': record_count,
                    'output_file': output_file
                }
            else:
                return {
                    'status': 'failed',
                    'runtime': runtime,
                    'stderr': result.stderr[-300:] if result.stderr else "",
                    'returncode': result.returncode
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'runtime': 0.0
            }
    
    def _estimate_ncp_from_result(self, result, original_data):
        """Estimate NCP from anonymized result (simplified calculation)"""
        if not result or len(result) == 0:
            return 100.0
        
        # Simple estimation based on generalized values
        # This is a rough approximation since we don't have exact NCP calculation
        
        total_loss = 0.0
        total_cells = 0
        
        for i, qi_idx in enumerate(self.qi_indices):
            if qi_idx not in self.categorical_indices:
                # Numeric attribute - check for ranges
                generalized_values = [row[qi_idx] for row in result]
                unique_generalized = set(generalized_values)
                
                for val in unique_generalized:
                    if ',' in str(val):  # Range like "25,30"
                        try:
                            parts = val.split(',')
                            if len(parts) == 2:
                                range_width = float(parts[1]) - float(parts[0])
                                # Get original range for normalization
                                original_values = [row[qi_idx] for row in original_data]
                                original_range = max(float(max(original_values)) - float(min(original_values)), 1)
                                loss = range_width / original_range
                                total_loss += loss
                        except (ValueError, IndexError):
                            total_loss += 0.5  # Assume moderate loss
                    # Single values contribute 0 loss
                    
                total_cells += 1
            else:
                # Categorical attribute - check for wildcards
                generalized_values = [row[qi_idx] for row in result]
                unique_generalized = set(generalized_values)
                
                wildcard_ratio = sum(1 for val in generalized_values if str(val) == '*') / len(generalized_values)
                total_loss += wildcard_ratio
                total_cells += 1
        
        if total_cells == 0:
            return 0.0
        
        ncp_percentage = (total_loss / total_cells) * 100
        return min(ncp_percentage, 100.0)  # Cap at 100%
    
    def _extract_ncp_from_output(self, output: str) -> float:
        """Extract NCP percentage from command output"""
        try:
            lines = output.split('\\n')
            for line in lines:
                if 'Information loss' in line and 'NCP' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        ncp_str = parts[-1].strip().replace('%', '')
                        return float(ncp_str)
                elif 'NCP' in line and '%' in line:
                    words = line.split()
                    for word in words:
                        if '%' in word:
                            return float(word.replace('%', ''))
        except (ValueError, IndexError):
            pass
        return 0.0
    
    def _count_records(self, csv_file: str) -> int:
        """Count records in CSV file"""
        try:
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    return sum(1 for row in reader)
        except Exception:
            pass
        return 0
    
    def run_comparison(self):
        """Main comparison workflow"""
        print("[START] Direct Algorithm Comparison")
        print(f"[FILE] Test file: {self.test_file}")
        print(f"[PARAMS] k={self.k}, QI={self.qi_indices}")
        
        if not os.path.exists(self.test_file):
            print(f"[ERROR] Test file {self.test_file} not found!")
            return
        
        # Load test data
        print("\n[LOADING] Test data...")
        data = self.load_test_data()
        print(f"[LOADED] {len(data)} records")
        
        # Build attribute trees
        print("[BUILDING] Attribute trees...")
        att_trees = self.build_attribute_trees(data)
        print(f"[BUILT] {len(att_trees)} attribute trees")
        
        results = {
            'original': {},
            'standalone': {}
        }
        
        # Run original implementations
        print("\n[RUNNING] Original Implementations...")
        
        # Mondrian
        results['original']['mondrian'] = self.run_original_mondrian(data, att_trees)
        
        # TDG  
        results['original']['tdg'] = self.run_original_tdg(data, att_trees)
        
        # CB variants
        for cb_type in ['knn', 'kmember', 'oka']:
            results['original'][f'cb_{cb_type}'] = self.run_original_cb(data, att_trees, cb_type)
        
        # Run standalone implementations
        print("\n[RUNNING] Standalone Implementations...")
        
        standalone_configs = {
            'mondrian': ('mondrian_standalone.py', []),
            'tdg': ('tdg_standalone.py', []),
            'cb_knn': ('cb_standalone.py', ['--cb-alg', 'knn']),
            'cb_kmember': ('cb_standalone.py', ['--cb-alg', 'kmember']),
            'cb_oka': ('cb_standalone.py', ['--cb-alg', 'oka'])
        }
        
        for alg_name, (script, extra_args) in standalone_configs.items():
            results['standalone'][alg_name] = self.run_standalone_algorithm(alg_name, script, extra_args)
        
        # Generate comparison report
        self.generate_comparison_report(results['original'], results['standalone'])
        
        # Cleanup temp files
        self.cleanup_temp_files()
    
    def generate_comparison_report(self, original_results: Dict, standalone_results: Dict):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("DIRECT ALGORITHM COMPARISON REPORT")
        print("="*80)
        
        # Summary table
        print(f"\n{'Algorithm':<15} {'Original':<20} {'Standalone':<20} {'NCP Diff':<12} {'Time Diff':<12}")
        print("-" * 85)
        
        algorithms = ['mondrian', 'tdg', 'cb_knn', 'cb_kmember', 'cb_oka']
        successful_comparisons = []
        
        for alg in algorithms:
            orig = original_results.get(alg, {})
            standalone = standalone_results.get(alg, {})
            
            # Format original results
            if orig.get('status') == 'success':
                orig_str = f"{orig.get('ncp', 0):.1f}% ({orig.get('runtime', 0):.2f}s)"
            else:
                orig_str = f"{orig.get('status', 'unknown')}"
            
            # Format standalone results
            if standalone.get('status') == 'success':
                standalone_str = f"{standalone.get('ncp', 0):.1f}% ({standalone.get('runtime', 0):.2f}s)"
            else:
                standalone_str = f"{standalone.get('status', 'unknown')}"
            
            # Calculate differences
            ncp_diff = ""
            time_diff = ""
            if (orig.get('status') == 'success' and standalone.get('status') == 'success'):
                ncp_delta = standalone.get('ncp', 0) - orig.get('ncp', 0)
                time_delta = standalone.get('runtime', 0) - orig.get('runtime', 0)
                ncp_diff = f"{ncp_delta:+.1f}%"
                time_diff = f"{time_delta:+.2f}s"
                
                successful_comparisons.append({
                    'algorithm': alg,
                    'ncp_delta': ncp_delta,
                    'time_delta': time_delta,
                    'orig_ncp': orig.get('ncp', 0),
                    'standalone_ncp': standalone.get('ncp', 0)
                })
            
            print(f"{alg:<15} {orig_str:<20} {standalone_str:<20} {ncp_diff:<12} {time_diff:<12}")
        
        # Detailed analysis
        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)
        
        for comparison in successful_comparisons:
            alg = comparison['algorithm']
            ncp_delta = comparison['ncp_delta']
            time_delta = comparison['time_delta']
            
            print(f"\n[ANALYSIS] {alg.upper()}")
            print("-" * 40)
            print(f"  [NCP] Original: {comparison['orig_ncp']:.2f}%, Standalone: {comparison['standalone_ncp']:.2f}%")
            print(f"  [DIFF] NCP: {ncp_delta:+.2f}% ({'worse' if ncp_delta > 0 else 'better'})")
            print(f"  [TIME] Difference: {time_delta:+.2f}s ({'slower' if time_delta > 0 else 'faster'})")
            
            # Quality assessment
            if abs(ncp_delta) <= 2.0:
                quality = "[EQUIVALENT]"
            elif abs(ncp_delta) <= 5.0:
                quality = "[ACCEPTABLE]"
            else:
                quality = "[SIGNIFICANT DIFFERENCE]"
            
            print(f"  [QUALITY] Assessment: {quality}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if successful_comparisons:
            avg_ncp_delta = sum(c['ncp_delta'] for c in successful_comparisons) / len(successful_comparisons)
            avg_time_delta = sum(c['time_delta'] for c in successful_comparisons) / len(successful_comparisons)
            
            print(f"[SUCCESS] Compared {len(successful_comparisons)}/{len(algorithms)} algorithms")
            print(f"[NCP] Average difference: {avg_ncp_delta:+.2f}%")
            print(f"[TIME] Average difference: {avg_time_delta:+.2f}s")
            
            if abs(avg_ncp_delta) <= 3.0:
                print("[OVERALL] Standalone implementations maintain equivalent quality!")
            else:
                print("[OVERALL] Some quality differences detected, review algorithms")
        else:
            print("[ERROR] No successful comparisons")
        
        # Save detailed results
        self._save_results({
            'original': original_results,
            'standalone': standalone_results,
            'summary': {
                'successful_comparisons': len(successful_comparisons),
                'avg_ncp_delta': avg_ncp_delta if successful_comparisons else 0,
                'avg_time_delta': avg_time_delta if successful_comparisons else 0
            }
        })
    
    def _save_results(self, results):
        """Save results to JSON file"""
        results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        results['test_parameters'] = {
            'k': self.k,
            'qi_indices': self.qi_indices,
            'sa_indices': self.sa_indices,
            'categorical_indices': list(self.categorical_indices),
            'test_file': self.test_file
        }
        
        with open('direct_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SAVED] Results saved to: direct_comparison_results.json")
    
    def cleanup_temp_files(self):
        """Remove temporary files"""
        temp_files = [
            'temp_standalone_mondrian_output.csv',
            'temp_standalone_tdg_output.csv',
            'temp_standalone_cb_knn_output.csv',
            'temp_standalone_cb_kmember_output.csv',
            'temp_standalone_cb_oka_output.csv'
        ]
        
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception:
                pass


def main():
    """Main entry point"""
    try:
        comparison = DirectAlgorithmComparison()
        comparison.run_comparison()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Comparison interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()