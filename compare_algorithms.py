#!/usr/bin/env python3
"""
Algorithm Comparison Script: Original vs Standalone Implementations

This script compares the performance and results of original and standalone
implementations of Mondrian, TDG, and CB k-anonymity algorithms on the 1000-row
adult subset dataset.

Usage:
    python compare_algorithms.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class AlgorithmComparison:
    """Handles comparison between original and standalone algorithm implementations"""
    
    def __init__(self):
        self.test_file = "adult_subset_1000.csv"
        self.results = {}
        self.k = 5
        self.qi_columns = "0,1,5,6,7,8,9"
        self.categorical = "1,5,6,7,8,9"
        
    def run_original_algorithms(self) -> Dict[str, Dict]:
        """Run original implementations via baseline script"""
        print("[RUNNING] Original Implementations...")
        results = {}
        
        algorithms = {
            'mondrian': 'mondrian',
            'tdg': 'tdg', 
            'cb_knn': 'cb',
            'cb_kmember': 'cb',
            'cb_oka': 'cb'
        }
        
        for alg_name, baseline_alg in algorithms.items():
            print(f"  Running original {alg_name}...")
            
            try:
                # Build command for baseline script using typer command structure
                cmd = [
                    sys.executable, "baseline_with_repetitions.py",
                    baseline_alg,  # Command first
                    "adult", "rf",  # Then positional args
                    "--start-k", str(self.k),
                    "--stop-k", str(self.k),
                    "--step-k", "1"
                ]
                
                # Add CB algorithm type for clustering-based methods
                if alg_name.startswith('cb_'):
                    cb_type = alg_name.split('_')[1]  # knn, kmember, or oka
                    cmd.extend(["--cb-alg", cb_type])
                
                # Run with timeout
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300,  # 5 minute timeout
                    cwd=os.getcwd()
                )
                runtime = time.time() - start_time
                
                if result.returncode == 0:
                    # Parse output for NCP and other metrics
                    ncp = self._extract_ncp_from_output(result.stdout)
                    results[alg_name] = {
                        'status': 'success',
                        'runtime': runtime,
                        'ncp': ncp,
                        'stdout': result.stdout[-500:],  # Last 500 chars
                        'stderr': result.stderr[-500:] if result.stderr else ""
                    }
                    print(f"    [SUCCESS] {runtime:.2f}s, NCP: {ncp:.2f}%")
                else:
                    results[alg_name] = {
                        'status': 'failed',
                        'runtime': runtime,
                        'stdout': result.stdout[-500:],
                        'stderr': result.stderr[-500:] if result.stderr else "",
                        'returncode': result.returncode
                    }
                    print(f"    [FAILED] code {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                results[alg_name] = {
                    'status': 'timeout',
                    'runtime': 300.0,
                    'error': 'Process timed out after 5 minutes'
                }
                print(f"    [TIMEOUT] >5 minutes")
            except Exception as e:
                results[alg_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    [ERROR] {e}")
        
        return results
    
    def run_standalone_algorithms(self) -> Dict[str, Dict]:
        """Run standalone implementations"""
        print("\n[RUNNING] Standalone Implementations...")
        results = {}
        
        algorithms = {
            'mondrian': {
                'script': 'mondrian_standalone.py',
                'extra_args': []
            },
            'tdg': {
                'script': 'tdg_standalone.py', 
                'extra_args': []
            },
            'cb_knn': {
                'script': 'cb_standalone.py',
                'extra_args': ['--cb-alg', 'knn']
            },
            'cb_kmember': {
                'script': 'cb_standalone.py',
                'extra_args': ['--cb-alg', 'kmember']
            },
            'cb_oka': {
                'script': 'cb_standalone.py',
                'extra_args': ['--cb-alg', 'oka']
            }
        }
        
        for alg_name, config in algorithms.items():
            print(f"  Running standalone {alg_name}...")
            
            try:
                # Build command
                output_file = f"temp_{alg_name}_output.csv"
                cmd = [
                    sys.executable, config['script'],
                    self.test_file, output_file,
                    "--k", str(self.k),
                    "--qi-columns", self.qi_columns,
                    "--categorical", self.categorical,
                    "--header"
                ] + config['extra_args']
                
                # Run with timeout
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=os.getcwd()
                )
                runtime = time.time() - start_time
                
                if result.returncode == 0:
                    # Parse output for metrics
                    ncp = self._extract_ncp_from_output(result.stdout)
                    record_count = self._extract_record_count(output_file)
                    
                    results[alg_name] = {
                        'status': 'success',
                        'runtime': runtime,
                        'ncp': ncp,
                        'records': record_count,
                        'output_file': output_file,
                        'stdout': result.stdout[-500:],
                        'stderr': result.stderr[-500:] if result.stderr else ""
                    }
                    print(f"    [SUCCESS] {runtime:.2f}s, NCP: {ncp:.2f}%, Records: {record_count}")
                else:
                    results[alg_name] = {
                        'status': 'failed',
                        'runtime': runtime,
                        'stdout': result.stdout[-500:],
                        'stderr': result.stderr[-500:] if result.stderr else "",
                        'returncode': result.returncode
                    }
                    print(f"    [FAILED] code {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                results[alg_name] = {
                    'status': 'timeout',
                    'runtime': 300.0,
                    'error': 'Process timed out after 5 minutes'
                }
                print(f"    [TIMEOUT] >5 minutes")
            except Exception as e:
                results[alg_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    [ERROR] {e}")
        
        return results
    
    def _extract_ncp_from_output(self, output: str) -> float:
        """Extract NCP percentage from command output"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'Information loss' in line and 'NCP' in line:
                    # Look for pattern like "Information loss (NCP): 42.91%"
                    parts = line.split(':')
                    if len(parts) > 1:
                        ncp_str = parts[-1].strip().replace('%', '')
                        return float(ncp_str)
                elif 'NCP' in line and '%' in line:
                    # Look for pattern like "NCP 42.91%"
                    words = line.split()
                    for word in words:
                        if '%' in word:
                            return float(word.replace('%', ''))
        except (ValueError, IndexError):
            pass
        return 0.0
    
    def _extract_record_count(self, output_file: str) -> int:
        """Count records in output CSV file"""
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    count = sum(1 for row in reader) - 1  # Subtract header
                    return count
        except Exception:
            pass
        return 0
    
    def generate_comparison_report(self, original_results: Dict, standalone_results: Dict):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON REPORT")
        print("="*80)
        
        # Summary table
        print(f"\n{'Algorithm':<15} {'Original':<15} {'Standalone':<15} {'NCP Diff':<12} {'Time Diff':<12}")
        print("-" * 75)
        
        for alg in ['mondrian', 'tdg', 'cb_knn', 'cb_kmember', 'cb_oka']:
            orig = original_results.get(alg, {})
            standalone = standalone_results.get(alg, {})
            
            # Format original results
            if orig.get('status') == 'success':
                orig_str = f"{orig.get('ncp', 0):.1f}% ({orig.get('runtime', 0):.1f}s)"
            else:
                orig_str = f"{orig.get('status', 'unknown')}"
            
            # Format standalone results
            if standalone.get('status') == 'success':
                standalone_str = f"{standalone.get('ncp', 0):.1f}% ({standalone.get('runtime', 0):.1f}s)"
            else:
                standalone_str = f"{standalone.get('status', 'unknown')}"
            
            # Calculate differences
            ncp_diff = ""
            time_diff = ""
            if (orig.get('status') == 'success' and standalone.get('status') == 'success'):
                ncp_delta = standalone.get('ncp', 0) - orig.get('ncp', 0)
                time_delta = standalone.get('runtime', 0) - orig.get('runtime', 0)
                ncp_diff = f"{ncp_delta:+.1f}%"
                time_diff = f"{time_delta:+.1f}s"
            
            print(f"{alg:<15} {orig_str:<15} {standalone_str:<15} {ncp_diff:<12} {time_diff:<12}")
        
        # Detailed analysis
        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)
        
        successful_comparisons = []
        
        for alg in ['mondrian', 'tdg', 'cb_knn', 'cb_kmember', 'cb_oka']:
            orig = original_results.get(alg, {})
            standalone = standalone_results.get(alg, {})
            
            print(f"\n[ANALYSIS] {alg.upper()}")
            print("-" * 40)
            
            if orig.get('status') == 'success' and standalone.get('status') == 'success':
                orig_ncp = orig.get('ncp', 0)
                standalone_ncp = standalone.get('ncp', 0)
                orig_time = orig.get('runtime', 0)
                standalone_time = standalone.get('runtime', 0)
                
                ncp_delta = standalone_ncp - orig_ncp
                time_delta = standalone_time - orig_time
                
                print(f"  [NCP] Comparison:")
                print(f"    Original:   {orig_ncp:.2f}%")
                print(f"    Standalone: {standalone_ncp:.2f}%")
                print(f"    Difference: {ncp_delta:+.2f}% ({'worse' if ncp_delta > 0 else 'better'})")
                
                print(f"  [TIME] Runtime Comparison:")
                print(f"    Original:   {orig_time:.2f}s")
                print(f"    Standalone: {standalone_time:.2f}s") 
                print(f"    Difference: {time_delta:+.2f}s ({'slower' if time_delta > 0 else 'faster'})")
                
                # Quality assessment
                if abs(ncp_delta) <= 2.0:
                    quality = "[EQUIVALENT]"
                elif ncp_delta <= 5.0:
                    quality = "[ACCEPTABLE]"
                else:
                    quality = "[SIGNIFICANT DIFFERENCE]"
                
                print(f"  [QUALITY] Assessment: {quality}")
                
                successful_comparisons.append({
                    'algorithm': alg,
                    'ncp_delta': ncp_delta,
                    'time_delta': time_delta,
                    'orig_ncp': orig_ncp,
                    'standalone_ncp': standalone_ncp
                })
                
            else:
                print(f"  [ERROR] Cannot compare:")
                print(f"    Original: {orig.get('status', 'unknown')}")
                print(f"    Standalone: {standalone.get('status', 'unknown')}")
                
                if orig.get('stderr'):
                    print(f"    Original Error: {orig['stderr'][:100]}...")
                if standalone.get('stderr'):
                    print(f"    Standalone Error: {standalone['stderr'][:100]}...")
        
        # Summary recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if successful_comparisons:
            avg_ncp_delta = sum(c['ncp_delta'] for c in successful_comparisons) / len(successful_comparisons)
            avg_time_delta = sum(c['time_delta'] for c in successful_comparisons) / len(successful_comparisons)
            
            print(f"[SUCCESS] Compared {len(successful_comparisons)}/5 algorithms")
            print(f"[NCP] Average difference: {avg_ncp_delta:+.2f}%")
            print(f"[TIME] Average difference: {avg_time_delta:+.2f}s")
            
            if abs(avg_ncp_delta) <= 3.0:
                print("[OVERALL] Standalone implementations maintain good quality!")
            else:
                print("[OVERALL] Some quality differences detected, review needed")
        else:
            print("[ERROR] No successful comparisons - check implementations")
        
        # Save detailed results
        self._save_detailed_results(original_results, standalone_results)
    
    def _save_detailed_results(self, original_results: Dict, standalone_results: Dict):
        """Save detailed results to JSON file"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_parameters': {
                'k': self.k,
                'qi_columns': self.qi_columns,
                'categorical': self.categorical,
                'test_file': self.test_file
            },
            'original_results': original_results,
            'standalone_results': standalone_results
        }
        
        with open('algorithm_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SAVED] Detailed results saved to: algorithm_comparison_results.json")
    
    def cleanup_temp_files(self):
        """Remove temporary output files"""
        temp_files = [
            'temp_mondrian_output.csv',
            'temp_tdg_output.csv', 
            'temp_cb_knn_output.csv',
            'temp_cb_kmember_output.csv',
            'temp_cb_oka_output.csv'
        ]
        
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception:
                pass
    
    def run_comparison(self):
        """Main comparison workflow"""
        print("[START] Algorithm Comparison")
        print(f"[FILE] Test file: {self.test_file}")
        print(f"[PARAMS] k={self.k}, QI={self.qi_columns}")
        
        # Check if test file exists
        if not os.path.exists(self.test_file):
            print(f"[ERROR] Test file {self.test_file} not found!")
            return
        
        try:
            # Run original implementations
            original_results = self.run_original_algorithms()
            
            # Run standalone implementations  
            standalone_results = self.run_standalone_algorithms()
            
            # Generate comparison report
            self.generate_comparison_report(original_results, standalone_results)
            
        finally:
            # Cleanup
            self.cleanup_temp_files()


def main():
    """Main entry point"""
    try:
        comparison = AlgorithmComparison()
        comparison.run_comparison()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Comparison interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()