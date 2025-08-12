# Top-Down Greedy (TDG) k-Anonymity Algorithm Analysis

## 📊 **Algorithm Overview**

The Top-Down Greedy (TDG) algorithm is a **heuristic partitioning algorithm** for k-anonymity that uses a greedy approach to recursively split data partitions. Unlike Mondrian which splits on dimensions, TDG splits partitions by finding pairs of records that are furthest apart and using them as "centroids" to distribute other records.

## 📥 **Input Parameters**

The main function `tdg_get_result_one()` takes these parameters:

1. **`att_trees`** - Attribute hierarchy trees for generalization
   - **Numeric attributes**: `NumRange` objects with sorted values, ranges, and support
   - **Categorical attributes**: Generalization hierarchy trees as dictionaries

2. **`data`** - Raw dataset as 2D list/array of records

3. **`k`** - Privacy parameter (minimum group size for k-anonymity)

4. **`path`** - File system path for data storage

5. **`qi_index`** - List of quasi-identifier column indices (sensitive columns to anonymize)

6. **`SA_index`** - List of sensitive attribute indices (preserved, not anonymized)

7. **`logger`** - Optional logger for structured output

## ⚙️ **Algorithm Workflow**

### 🔄 **Core Algorithm Steps:**

1. **Initialization (`init()`):**
   ```python
   # Set global variables
   GL_K = k                    # k-anonymity parameter
   QI_LEN = len(qi_index)     # Number of quasi-identifiers
   ATT_TREES = att_trees      # Attribute hierarchies
   IS_CAT = [...]             # Boolean array: True=categorical, False=numeric
   ROUNDS = 3                 # Number of iterations for pair selection
   ```

2. **Create Initial Partition:**
   ```python
   # Create single partition containing all records
   whole_partition = Partition(data, middle)
   # middle: generalization level (starts at most general)
   ```

3. **Recursive Partitioning (`anonymize()`):**
   ```python
   def anonymize(partition):
       if not can_split(partition):
           RESULT.append(partition)  # Add to final result
           return
       
       u, v = get_pair(partition)                    # Find furthest pair
       sub_partitions = distribute_record(u, v, partition)  # Split based on pair
       
       # Ensure k-anonymity constraint
       if len(sub_partitions[0]) < GL_K:
           balance(sub_partitions, 0)
       elif len(sub_partitions[1]) < GL_K:
           balance(sub_partitions, 1)
       
       for sub_partition in sub_partitions:
           anonymize(sub_partition)  # Recursively process sub-partitions
   ```

### 📐 **Pair Selection (`get_pair()`):**
- **Finds two records that are furthest apart in the partition**
- **Uses multiple rounds (ROUNDS=3) to find optimal pairs**
- **Distance calculation considers both numeric and categorical attributes**
- **Serves as "centroids" for binary partitioning**

### ✂️ **Record Distribution (`distribute_record()`):**
- **Assigns each record to the partition of its closest centroid (u or v)**
- **Uses same distance function as pair selection**
- **Creates two sub-partitions based on similarity to centroids**

### 🔄 **Balancing (`balance()`):**
- **Ensures both partitions have at least k records**
- **Moves records from larger partition to smaller one**
- **Chooses records that minimize information loss**

### 🛑 **Termination Conditions:**
- Partition size < 2k (cannot split while maintaining k-anonymity)
- `can_split()` returns false (no valid splits possible)

## 📈 **Key Data Structures**

### **`Partition` Class:**
```python
class Partition:
    member: []        # Records in this partition
    middle: []        # Generalization level for each QI
    can_split: bool   # Whether partition can be further split
```

### **Distance Calculation:**
- **Numeric attributes**: Euclidean distance in normalized space
- **Categorical attributes**: 0 if same, 1 if different
- **Combined metric**: Weighted sum of numeric and categorical distances

## 📊 **Output & Metrics**

### **Returns:**
1. **Anonymized Dataset**: 2D array with generalized values
2. **Performance Metrics**:
   - **NCP (Normalized Certainty Penalty)**: Information loss percentage
   - **DP (Discernability Penalty)**: Sum of squared partition sizes
   - **Runtime**: Processing time in seconds

### **NCP Calculation:**
```python
# For each partition:
rncp = NCP(generalization_result)
rncp *= partition_size
total_ncp += rncp
# Final NCP = (total_ncp / total_records / num_QI_attributes) * 100
```

## 🔍 **Algorithm Characteristics**

### ✅ **Strengths:**
- **Better Partitioning**: Uses centroids instead of arbitrary dimension splits
- **Flexible**: Works with mixed numeric/categorical data
- **Balancing**: Explicit balancing mechanism ensures k-anonymity
- **Distance-based**: More sophisticated record grouping than simple splits

### ⚠️ **Limitations:**
- **Computationally Expensive**: Distance calculations for all record pairs
- **Greedy**: No backtracking or global optimization
- **Parameter Sensitive**: ROUNDS parameter affects quality vs. performance
- **Heuristic**: No theoretical optimality guarantees

## 🎨 **Visual Concept**
Unlike Mondrian's rectangular partitions, TDG creates more natural clusters by grouping records around two "centroid" records that are maximally distant. This can lead to better utility preservation as records are grouped by similarity rather than arbitrary dimensional cuts.

## 🏗️ **Implementation Details**

### **File Structure:**
- `top_down_greedy/top_down_greedy_anonymization.py` - Core algorithm
- `top_down_greedy/anonymizer.py` - Entry point and wrapper functions

### **Key Functions:**
- `Top_Down_Greedy_Anonymization()` - Main algorithm orchestrator
- `anonymize()` - Recursive partitioning function
- `get_pair()` - Find furthest record pair
- `distribute_record()` - Assign records to partitions
- `balance()` - Ensure k-anonymity constraint

### **Algorithm Complexity:**
- **Time**: O(n² × log n) due to pairwise distance calculations
- **Space**: O(n) for storing partitions and results

## 📚 **Usage Example**

```python
from top_down_greedy.anonymizer import tdg_get_result_one

# Parameters
att_trees = [...]      # Attribute hierarchy trees
data = [...]          # Raw dataset
k = 5                 # k-anonymity parameter
qi_index = [0,1,2]    # Quasi-identifier columns
SA_index = [3]        # Sensitive attribute columns

# Run algorithm
result = tdg_get_result_one(att_trees, data, k, ".", qi_index, SA_index)
anonymized_data = result  # Returns generalized dataset
```

## 🔍 **Key Differences from Mondrian**

| Aspect | Mondrian | TDG |
|--------|----------|-----|
| **Splitting Strategy** | Dimension-based (largest width) | Distance-based (furthest pair) |
| **Partitions** | Rectangular regions | Natural clusters |
| **Complexity** | O(n log n) | O(n² log n) |
| **Quality** | Good for most cases | Better for clustered data |
| **Balance** | Implicit through median | Explicit balancing step |

## 🔗 **References**

This implementation is based on:
- Top-Down Greedy Anonymization research
- Implementation by Qiyuan Gong (migrated from Python 2 to Python 3)
- Paper: "k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers"

The algorithm is particularly effective for datasets where records naturally cluster in the quasi-identifier space, as it can create more coherent groupings than purely dimension-based approaches.