# Basic Mondrian k-Anonymity Algorithm Analysis

## 📊 **Algorithm Overview**

The Basic Mondrian algorithm is a **top-down, greedy partitioning algorithm** for k-anonymity that works by recursively splitting data into partitions until each partition contains at least k records. It's inspired by Mondrian's art style of rectangular partitioning.

## 📥 **Input Parameters**

The main function `get_result_one()` takes these parameters:

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
   ```

2. **Create Initial Partition:**
   ```python
   # Create single partition containing all records
   whole_partition = Partition(data, width, middle)
   # width: domain ranges for each QI attribute
   # middle: generalization level (starts at most general)
   ```

3. **Recursive Partitioning (`anonymize()`):**
   ```python
   def anonymize(partition):
       if not check_splitable(partition):
           RESULT.append(partition)  # Add to final result
           return
       
       dim = choose_dimension(partition)     # Pick dimension to split
       sub_partitions = split_partition(partition, dim)
       
       for sub_p in sub_partitions:
           anonymize(sub_p)  # Recursively process sub-partitions
   ```

### 📐 **Dimension Selection (`choose_dimension()`):**
- **Chooses the dimension with the largest normalized width**
- **Normalized width calculation:**
  - **Numeric**: `(high_value - low_value) / total_range`
  - **Categorical**: `num_categories / total_categories`

### ✂️ **Partition Splitting:**

**For Numeric Attributes:**
- Find median value that splits records roughly in half
- Ensure each split has ≥ k records
- Create range-based generalizations (e.g., "25,35")

**For Categorical Attributes:**
- Use generalization hierarchy tree
- Split based on categorical groupings
- Create hierarchy-based generalizations (e.g., "Manager" → "Professional")

### 🛑 **Termination Conditions:**
- Partition size < 2k (cannot split while maintaining k-anonymity)
- No splittable dimensions remain
- All dimensions have been exhausted

## 📈 **Key Data Structures**

### **`Partition` Class:**
```python
class Partition:
    member: []     # Records in this partition
    width: []      # Domain width for each QI attribute
    middle: []     # Generalization level for each QI
    allow: []      # Which dimensions can still be split
```

### **Attribute Trees:**
- **`NumRange`**: For numeric attributes (age, salary)
  - `sort_value`: Sorted unique values
  - `range`: Total numeric range
  - `support`: Value frequencies

- **Categorical Trees**: Hierarchical dictionaries
  - Tree structure for generalization levels
  - E.g., "Doctor" → "Professional" → "Worker" → "*"

## 📊 **Output & Metrics**

### **Returns:**
1. **Anonymized Dataset**: 2D array with generalized values
2. **Performance Metrics**:
   - **NCP (Normalized Certainty Penalty)**: Information loss percentage
   - **Runtime**: Processing time in seconds

### **NCP Calculation:**
```python
# For each partition and QI attribute:
ncp += (normalized_width * partition_size)
# Final NCP = (total_ncp / num_QI_attributes / total_records) * 100
```

## 🔍 **Algorithm Characteristics**

### ✅ **Strengths:**
- **Efficient**: O(n log n) time complexity
- **Simple**: Greedy approach, easy to understand
- **Flexible**: Handles both numeric and categorical data
- **Quality**: Often produces good utility/privacy trade-offs

### ⚠️ **Limitations:**
- **Greedy**: May not find globally optimal partitions
- **Dimension Bias**: Tends to split on high-cardinality attributes
- **No Backtracking**: Cannot undo poor early splits
- **Rectangle Constraint**: Partitions must be rectangular in attribute space

## 🎨 **Visual Concept**
Think of Mondrian's rectangular paintings - the algorithm creates non-overlapping rectangular regions in the multi-dimensional attribute space, where each rectangle contains ≥ k records and all records in a rectangle get the same generalized values.

## 🏗️ **Implementation Details**

### **File Structure:**
- `basic_mondrian/mondrian.py` - Core algorithm implementation
- `basic_mondrian/anonymizer.py` - Entry point and wrapper functions
- `basic_mondrian/models/numrange.py` - Numeric range data structure
- `basic_mondrian/utils/` - Utility functions for data handling

### **Key Functions:**
- `mondrian()` - Main algorithm orchestrator
- `anonymize()` - Recursive partitioning function
- `choose_dimension()` - Dimension selection heuristic
- `split_partition()` - Partition splitting logic
- `find_median()` - Median finding for numeric splits

## 📚 **Usage Example**

```python
from basic_mondrian.anonymizer import get_result_one

# Parameters
att_trees = [...]      # Attribute hierarchy trees
data = [...]          # Raw dataset
k = 5                 # k-anonymity parameter
qi_index = [0,1,2]    # Quasi-identifier columns
SA_index = [3]        # Sensitive attribute columns

# Run algorithm
result = get_result_one(att_trees, data, k, ".", qi_index, SA_index)
anonymized_data = result  # Returns generalized dataset
```

## 🔗 **References**

This implementation is based on:
- Original Mondrian algorithm by Kristen LeFevre et al.
- Implementation by Qiyuan Gong (migrated from Python 2 to Python 3)
- Paper: "k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers"

The algorithm is particularly effective for datasets with mixed numeric/categorical quasi-identifiers and provides a good balance between implementation simplicity and anonymization quality.