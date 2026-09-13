
# Day 4 — Customer Behavior Clustering

## Overview

Unsupervised learning project that segments customers into distinct behavioral groups using K-Means clustering. Unlike supervised learning, there is no target column — the goal is to discover hidden structure in the data and make it interpretable.

---

## Dataset

**Marketing Campaign Dataset** (`marketing_campaign.csv`)
- **Source**: Kaggle
- **Size**: 2,240 customers × 29 features
- **Type**: CRM / marketing data
- **Features**: Demographics (age, income, education, marital status), product spending across 6 categories, purchase channel behavior, campaign acceptance history

---

## Workflow

```
Data Loading → EDA → Preprocessing → Feature Engineering
→ Scaling → Outlier Capping → Clustering → Evaluation → Visualization
```

---

## Key Steps

### 1. EDA
- Identified 24 missing values in `Income` (~1% of data)
- Detected `Z_CostContact` and `Z_Revenue` as zero-variance columns (single unique value each)
- No duplicate rows found

### 2. Preprocessing
- **Dropped**: `ID` (identifier), `Z_CostContact`, `Z_Revenue` (zero variance, no information)
- **Missing values**: Dropped the 24 rows with missing `Income` — imputation was avoided to prevent distorting the spending patterns central to clustering
- **Excluded from clustering**: `Response` (target-like label, should not influence grouping)

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `Age` | Derived from `Year_Birth` (2026 - Year_Birth) |
| `Customer_Tenure_Days` | Years since enrollment (2026 - enrollment year) |
| `Total_Spending` | Sum of all 6 product spend columns |

### 4. Encoding
- `Marital_Status` → One-hot encoded (no ordinal relationship between categories)
- `Education` → Ordinal encoded 0–4 (Basic → PhD), preserving the natural level order

### 5. Scaling
- Applied `StandardScaler` to all clustering features
- Required because K-Means uses Euclidean distance — without scaling, high-range features like `Income` would dominate over low-range features like `Kidhome`

### 6. Outlier Handling
- Detected outliers using IQR method on 14 continuous numeric columns
- Applied **capping (clip)** at IQR bounds rather than dropping rows — preserves genuine high-value customers while limiting the distorting effect of extreme values

### 7. Choosing K
- **Elbow Method**: No sharp elbow — the inertia curve declined smoothly across K=2 to K=10
- **Silhouette Score**: Highest at K=2, confirming it as the best data-driven choice
- **Final decision**: K=2 — supported by both the Silhouette Score and a clear business interpretation (two natural customer segments)

### 8. Clustering
- Algorithm: **K-Means** (`n_clusters=2`, `n_init=10`, `random_state=42`)
- Also experimented with removing `Total_Spending` (collinear with individual spend columns) to verify cluster stability

### 9. Visualization
- **PCA** used to compress features into 2 dimensions for plotting
- PCA was applied **after** clustering, purely for visualization — clustering was performed on the full scaled feature space
- PC1: ~20%, PC2: ~7% (total 27% explained variance in 2D)
- Convex hull boundaries drawn around each cluster

---

## Results

| Cluster | Label | Size |
|---|---|---|
| 0 | Low-Value Customers | ~1,312 |
| 1 | High-Value Customers | ~904 |

### Cluster Profiles

| Feature | Cluster 0 (Low-Value) | Cluster 1 (High-Value) |
|---|---|---|
| Income | ~38,700 | ~71,900 |
| Total Spending | ~180 | ~1,228 |
| Kidhome | 0.70 | 0.06 |
| NumStorePurchases | 3.9 | 8.6 |
| NumCatalogPurchases | 0.86 | 5.3 |
| NumWebVisitsMonth | 6.4 | 3.7 |
| Campaign Response Rate | ~10% | ~23% |

### Interpretation
- **Cluster 0 — Low-Value Customers**: Lower income, more children at home, low spending across all categories. Visit the website frequently but convert less. Less responsive to campaigns.
- **Cluster 1 — High-Value Customers**: Higher income, fewer children, significantly higher spending across all product categories. More active across store and catalog channels. More than twice as likely to respond to marketing campaigns.

---

## Concepts Practiced

- Unsupervised learning (no target label)
- K-Means clustering
- Elbow Method and Silhouette Score for K selection
- PCA for dimensionality reduction and visualization
- IQR-based outlier capping
- Ordinal and one-hot encoding
- StandardScaler (mandatory before distance-based algorithms)
- Cluster profiling and business interpretation

---

## Files

| File | Description |
|---|---|
| `DAY04_Customer_Behavior_Clustering_v2.ipynb` | Main notebook |
| `marketing_campaign.csv` | Dataset (Kaggle) |
| `DAY04_README.md` | This file |
