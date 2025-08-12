# Clustering-Based (CB) k-Anonymity Algorithm Analysis

## 📊 **Algorithm Overview**

The Clustering-Based (CB) algorithm uses **clustering techniques** to group similar records together for k-anonymity. Instead of recursive partitioning, it creates clusters of records and then generalizes all records within each cluster to the same values. The algorithm supports three different clustering strategies: k-NN, k-Member, and OKA.

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

7. **`type_alg`** - Clustering algorithm type: 'knn', 'kmember', or 'oka'

8. **`logger`** - Optional logger for structured output

## ⚙️ **Algorithm Workflow**

### 🔄 **Core Algorithm Steps:**

1. **Initialization (`init()`):**
   ```python
   # Set global variables
   ATT_TREES = att_trees      # Attribute hierarchies
   LEN_DATA = len(data)       # Dataset size
   QI_LEN = len(qi_index)     # Number of quasi-identifiers
   IS_CAT = [...]             # Boolean array: True=categorical, False=numeric
   LCA_CACHE = []             # Least Common Ancestor cache
   NCP_CACHE = {}             # NCP calculation cache
   ```

2. **Select Clustering Algorithm:**
   ```python
   if type_alg == 'knn':
       clusters = clustering_knn(data, k)
   elif type_alg == 'kmember':
       clusters = clustering_kmember(data, k)
   elif type_alg == 'oka':
       clusters = clustering_oka(data, k)
   ```

3. **Generate Final Results:**
   ```python
   # For each cluster, generalize all records to cluster.gen_result
   for cluster in clusters:
       for record in cluster.member:
           result.append(cluster.gen_result + sensitive_attributes)
   ```

## 🎯 **Clustering Strategies**

### **1. k-NN Clustering (`clustering_knn()`):**
- **Starts with each record as a separate cluster**
- **Iteratively merges closest clusters until all have ≥ k members**
- **Uses nearest neighbor distance for merging decisions**
- **Distance based on generalization cost (NCP)**

### **2. k-Member Clustering (`clustering_kmember()`):**
- **Creates clusters with exactly k members each**
- **Assigns records to minimize overall information loss**
- **More balanced cluster sizes than k-NN**
- **May have one cluster with > k members (remainder)**

### **3. OKA Clustering (`clustering_oka()`):**
- **One-pass K-means Anonymization**
- **More sophisticated centroid-based approach**
- **Updates cluster centers dynamically**
- **Better for larger datasets**

## 📈 **Key Data Structures**

### **`Cluster` Class:**
```python
class Cluster:
    member: []              # Records in this cluster
    gen_result: []          # Generalized values for cluster
    center: []              # Cluster centroid (for numeric attributes)
    information_loss: float # NCP for this cluster
```

### **Distance & Generalization:**
- **Distance Calculation**: Based on generalization hierarchy distance
- **LCA (Least Common Ancestor)**: For categorical attribute generalization
- **NCP Caching**: Reduces computational overhead for repeated calculations

## 🔄 **Detailed Clustering Process**

### **k-NN Process:**
```python
def clustering_knn(data, k):
    # 1. Initialize each record as individual cluster
    clusters = [Cluster([record], [record]) for record in data]
    
    # 2. Iteratively merge closest clusters
    while min(len(cluster) for cluster in clusters) < k:
        # Find two closest clusters
        merge_cluster, merge_index = find_closest_clusters(clusters)
        # Merge clusters and update generalization
        merge_clusters(clusters, merge_cluster, merge_index)
    
    return clusters
```

### **Information Loss Calculation:**
- **Numeric Attributes**: Range expansion cost
- **Categorical Attributes**: Hierarchy climbing cost
- **Overall NCP**: Weighted sum across all QI attributes

## 📊 **Output & Metrics**

### **Returns:**
1. **Anonymized Dataset**: 2D array with generalized cluster values
2. **Performance Metrics**:
   - **NCP (Normalized Certainty Penalty)**: Information loss percentage
   - **Runtime**: Processing time in seconds

### **NCP Calculation:**
```python
# Sum information loss across all clusters
total_ncp = sum(cluster.information_loss for cluster in clusters)
# Normalize by dataset size and QI count
ncp = (total_ncp / len(data) / QI_len) * 100
```

## 🔍 **Algorithm Characteristics**

### ✅ **Strengths:**
- **Natural Grouping**: Creates clusters based on record similarity
- **Multiple Strategies**: Choice of clustering algorithms for different data types
- **Caching**: Efficient NCP and LCA caching reduces computation time
- **Quality**: Often produces lower information loss than partitioning methods
- **Flexibility**: Works well with various data distributions

### ⚠️ **Limitations:**
- **Computational Cost**: Distance calculations can be expensive
- **Memory Usage**: Caching structures require additional memory
- **Parameter Sensitivity**: Performance varies with clustering strategy choice
- **Scalability**: May not scale well to very large datasets
- **Determinism**: Results may vary between runs (especially OKA)

## 🎨 **Visual Concept**
Unlike partition-based methods that create rigid boundaries, clustering creates natural "blobs" of similar records. Each blob becomes a cluster where all records are generalized to the same representative values, preserving natural data relationships.

## 🏗️ **Implementation Details**

### **File Structure:**
- `clustering_based/clustering_based_k_anon.py` - Core clustering algorithms
- `clustering_based/anonymizer.py` - Entry point and wrapper functions

### **Key Functions:**
- `clustering_based_k_anon()` - Main algorithm orchestrator
- `clustering_knn()` - k-Nearest Neighbor clustering
- `clustering_kmember()` - k-Member clustering
- `clustering_oka()` - One-pass K-means clustering
- `NCP()` - Information loss calculation
- `LCA()` - Least Common Ancestor for categorical generalization

### **Caching Mechanisms:**
- **LCA_CACHE**: Stores least common ancestor calculations
- **NCP_CACHE**: Caches normalized certainty penalty calculations
- **Significant Performance**: Can reduce runtime by 50-70%

## 📚 **Usage Example**

```python
from clustering_based.anonymizer import get_result_one

# Parameters
att_trees = [...]      # Attribute hierarchy trees
data = [...]          # Raw dataset
k = 5                 # k-anonymity parameter
qi_index = [0,1,2]    # Quasi-identifier columns
SA_index = [3]        # Sensitive attribute columns
type_alg = 'knn'      # Clustering algorithm: 'knn', 'kmember', or 'oka'

# Run algorithm
result = get_result_one(att_trees, data, k, ".", qi_index, SA_index, type_alg)
anonymized_data = result  # Returns generalized dataset
```

## 🔍 **Clustering Strategy Comparison**

| Strategy | Speed | Quality | Balance | Best For |
|----------|-------|---------|---------|----------|
| **k-NN** | Slow | High | Variable | Small datasets, high quality |
| **k-Member** | Medium | Medium | Good | Balanced performance |
| **OKA** | Fast | Medium | Good | Large datasets |

## 🎯 **When to Use CB Algorithm**

**Choose Clustering-Based when:**
- Data has natural cluster structures
- Want to preserve record relationships
- Have computational resources for distance calculations
- Need flexible cluster sizes
- Data quality is more important than speed

**Avoid when:**
- Very large datasets (>100k records)
- Memory is constrained
- Need deterministic results
- Simple rectangular partitions are sufficient

## 🔗 **References**

This implementation is based on:
- Clustering-based k-anonymity research
- Implementation by Qiyuan Gong (migrated from Python 2 to Python 3)  
- Paper: "k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers"

The algorithm is particularly effective for datasets with natural clustering patterns in the quasi-identifier space, often producing higher utility than partition-based methods at the cost of increased computational complexity.