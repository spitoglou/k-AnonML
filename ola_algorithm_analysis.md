# Optimal Lattice Anonymization (OLA) k-Anonymity Algorithm Analysis

## 📊 **Algorithm Overview**

The Optimal Lattice Anonymization (OLA) algorithm, also known as **ELEmam algorithm**, uses a **lattice-based approach** to find optimal generalization levels for k-anonymity. Instead of partitioning records, it explores the generalization lattice systematically to find the best combination of generalization levels that minimize information loss while satisfying k-anonymity.

## 📥 **Input Parameters**

The main function `main()` (called as `emain()`) takes these parameters:

1. **`raw_data`** - Raw dataset as 2D list/array of records

2. **`kanon`** - k-anonymity parameter (minimum group size)

3. **`gen_strat`** - Generalization strategies for each QI attribute
   - Functions that define how to generalize values at different levels
   - Can be simple functions or complex hierarchical strategies

4. **`max_gen_level`** - Maximum generalization level for each QI attribute
   - Defines the depth of the generalization lattice

5. **`qi_index`** - List of quasi-identifier column indices

6. **`metric`** - Optimization metric for selecting best nodes:
   - `'prec'` - Precision-based metric
   - `'gweight'` - Generalized weight metric  
   - `'aecs'` - Average Equivalence Class Size
   - `'dm'` - Discernability Metric
   - `'ent'` - Entropy-based metric
   - `'none'` - No specific metric (first valid solution)

7. **`res_folder`** - Results folder for output files

8. **`suppression_rate`** - Allowed suppression rate (0-100%)

## ⚙️ **Algorithm Workflow**

### 🔄 **Core Algorithm Steps:**

1. **Lattice Construction:**
   ```python
   # Create generalization lattice starting from root node
   rootnode = Node([0] * quasi_ident_count, node_array, max_gen_level, level_nodes)
   # Each node represents a generalization level combination
   # e.g., [0,1,2] means QI1 at level 0, QI2 at level 1, QI3 at level 2
   ```

2. **Systematic Exploration:**
   ```python
   # Sort nodes by lattice height (bottom-up evaluation)
   sorted_array = sort(node_array)
   
   # Find minimal k-anonymous nodes using ELEmam algorithm
   min_k = calculate(sorted_array, anonymity_checker)
   ```

3. **Best Node Selection:**
   ```python
   # Select best node based on chosen metric
   best_nodes = []
   for node in min_k:
       if metric == "prec" and node.prec < current_best:
           best_nodes = [node]
       elif metric == "gweight" and node.level < current_level:
           best_nodes = [node]
       # ... other metrics
   ```

4. **Data Anonymization:**
   ```python
   # Apply selected generalization levels to create anonymized data
   anon_data = create_anon_data(best_nodes, raw_data, qi_index, gen_strat, kanon, res_folder)
   ```

## 🌐 **Generalization Lattice Concept**

### **Lattice Structure:**
```
Level 2: [2,2,2] (most general)
       ↙   ↓   ↘
Level 1: [1,2,2] [2,1,2] [2,2,1]
       ↙   ↓   ↘   ↙   ↓   ↘
Level 0: [0,1,2] [1,0,2] [1,1,1] [0,2,1] [1,2,0] [2,0,1]
       ↙   ↓   ↘
Level -1: [0,0,2] [0,1,1] [1,0,1]
       ↙   ↓   ↘
Level -2: [0,0,1] [0,1,0] [1,0,0]
         ↙   ↓   ↘
Level -3: [0,0,0] (most specific)
```

### **Node Properties:**
- **`attributes`**: Generalization level for each QI [level1, level2, ...]
- **`prec`**: Precision metric value
- **`level`**: Total generalization level (sum of all QI levels)
- **`height`**: Position in lattice hierarchy

## 🔍 **ELEmam Algorithm Core**

### **Bottom-Up Evaluation:**
```python
def calculate(sorted_array, anonymity_checker):
    minimal_nodes = []
    
    for node in sorted_array:  # Bottom-up order
        # Check if node satisfies k-anonymity
        if anonymity_checker.is_k_anonymous(node):
            # Check if node is minimal (no ancestor already satisfies k-anonymity)
            if not has_k_anonymous_ancestor(node):
                minimal_nodes.append(node)
                # Mark all descendants as non-minimal
                mark_descendants(node)
    
    return minimal_nodes
```

### **k-Anonymity Checking:**
- **Generalize data** according to node's generalization levels
- **Group equivalent records** (records with same generalized values)
- **Count group sizes** and verify all groups have ≥ k members
- **Apply suppression** if allowed and beneficial

## 📈 **Optimization Metrics**

### **Available Metrics:**

1. **Precision (`prec`)**: Measures generalization precision
   - Lower values = better (more specific generalizations)

2. **Generalized Weight (`gweight`)**: Sum of generalization levels
   - Lower values = better (less generalization)

3. **Average Equivalence Class Size (`aecs`)**: 
   - Average size of equivalent record groups
   - Higher values = better (fewer, larger groups)

4. **Discernability Metric (`dm`)**: 
   - Penalty based on record distinguishability
   - Lower values = better

5. **Entropy (`ent`)**: Information-theoretic measure
   - Lower entropy = better (more uniform distribution)

## 📊 **Key Data Structures**

### **`Node` Class:**
```python
class Node:
    attributes: []      # Generalization levels [qi1_level, qi2_level, ...]
    prec: float        # Precision metric value
    level: int         # Sum of generalization levels
    height: int        # Lattice height
    children: []       # Child nodes in lattice
    parents: []        # Parent nodes in lattice
```

### **`AnonCheck` Class:**
```python
class AnonCheck:
    qi_data: [][]          # Quasi-identifier data
    max_gen_level: []      # Maximum generalization per QI
    gen_strat: []          # Generalization strategies
    allowed_suppressed: int # Maximum records to suppress
    kanon: int            # k-anonymity parameter
```

## 📊 **Output & Metrics**

### **Returns:**
1. **Anonymized Dataset**: 2D array with generalized values
2. **Generalization Array**: Selected generalization levels for each QI
3. **Suppression Information**: Number of suppressed records (if applicable)

### **Files Generated:**
- `genarray.csv` - Generalization levels used
- `supprarray.csv` - Suppression counts per iteration
- Anonymized data files in specified format

## 🔍 **Algorithm Characteristics**

### ✅ **Strengths:**
- **Optimal Search**: Systematic exploration of generalization space
- **Multiple Metrics**: Various optimization criteria available
- **Suppression Support**: Can suppress records instead of over-generalizing
- **Global Optimization**: Finds globally optimal solutions within lattice
- **Theoretical Foundation**: Based on solid mathematical framework
- **Flexible**: Works with any generalization hierarchy

### ⚠️ **Limitations:**
- **Exponential Complexity**: Lattice size grows exponentially with QI count
- **Memory Intensive**: Stores entire lattice structure
- **Scalability**: Limited to datasets with few QI attributes (typically ≤ 6)
- **Setup Complexity**: Requires defining generalization strategies and hierarchies
- **Runtime**: Can be very slow for large lattices

## 🎨 **Visual Concept**
Imagine a multi-dimensional lattice where each dimension represents a QI attribute and each point represents a possible generalization combination. OLA systematically explores this lattice from bottom (specific) to top (general) to find the optimal point that balances privacy and utility.

## 🏗️ **Implementation Details**

### **File Structure:**
- `elemam/main.py` - Main algorithm implementation
- `elemam/algorithm.py` - Core lattice calculation functions
- `elemam/node.py` - Node structure definition
- `elemam/kanon.py` - k-anonymity checking functions

### **Key Functions:**
- `main()` - Algorithm orchestrator and entry point
- `calculate()` - Core ELEmam lattice exploration
- `sort()` - Sort nodes by lattice height
- `create_anon_data()` - Generate anonymized dataset
- `AnonCheck.is_k_anonymous()` - Verify k-anonymity constraint

### **Complexity Analysis:**
- **Time**: O(2^(sum of max_gen_levels)) for lattice exploration
- **Space**: O(2^(sum of max_gen_levels)) for storing lattice
- **Practical Limit**: Usually ≤ 6 QI attributes with moderate generalization levels

## 📚 **Usage Example**

```python
from elemam.main import main as emain

# Parameters
raw_data = [...]              # Raw dataset
k = 5                        # k-anonymity parameter
gen_strat = [age_generalize, income_generalize, ...]  # Generalization functions
max_gen_level = [3, 4, 2]    # Max levels per QI
qi_index = [0, 1, 2]         # QI column indices
metric = 'gweight'           # Optimization metric
res_folder = "./results"     # Output folder
suppression_rate = 5         # Allow 5% suppression

# Run algorithm
anon_data, gen_levels = emain(raw_data, k, gen_strat, max_gen_level, 
                             qi_index, metric, res_folder, suppression_rate)
```

## 🎯 **When to Use OLA Algorithm**

**Choose OLA when:**
- Need provably optimal generalization levels
- Have few QI attributes (≤ 6)
- Can define clear generalization hierarchies
- Want to use suppression strategically
- Quality is more important than speed
- Need to compare different generalization strategies

**Avoid when:**
- Many QI attributes (> 6)
- Large datasets (> 50k records)
- Need fast results
- Don't have clear generalization hierarchies
- Memory is constrained

## 🔗 **References**

This implementation is based on:
- ELEmam (Optimal Lattice Anonymization) algorithm
- Research on lattice-based k-anonymity optimization
- Paper: "k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers"

The algorithm provides the theoretical foundation for optimal k-anonymity solutions, making it valuable for research and scenarios where finding the best possible generalization is crucial, despite computational constraints.