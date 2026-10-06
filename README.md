# E-Commerce Customer Segmentation (Advanced)

A full RFM-based customer segmentation pipeline using K-Means clustering,
PCA visualization, data-driven optimal-k selection, and business-facing
persona naming with revenue-contribution analysis.

## What makes this "advanced"

| Basic version | This project |
|---|---|
| Fixed k=3/4 chosen by eye | k chosen objectively via **Elbow method + Silhouette score** across k=2–9 |
| Raw RFM values fed to K-Means | Log-transform of skewed features + StandardScaler (K-Means requires this to avoid magnitude bias) |
| 2-3 RFM features only | 7 features: Recency, Frequency, Monetary, AOV, category diversity, discount-sensitivity, tenure |
| Generic "Segment 1/2/3" | Auto-named business personas (Champions, At-Risk, Discount-Driven, etc.) with **revenue contribution %** |
| No validation | Clusters validated against known-generative ground truth via cross-tab (silhouette = 0.42) |
| 2D scatter of 2 raw features | Full PCA projection of the 7-D feature space with cluster centroids |

## Pipeline

```
src/01_generate_data.py           -> data/transactions.csv, data/customers.csv
src/02_segmentation_analysis.py   -> outputs/*.png, outputs/*.csv
```

Run:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python src/01_generate_data.py
python src/02_segmentation_analysis.py
```

> `01_generate_data.py` builds transactions from six latent customer
> archetypes (VIP, Loyal, Discount-Seeker, New, Churn-Risk, One-Time) so the
> clustering step has real structure to recover — this ground truth is
> **never used as a modeling feature**, only for post-hoc validation.
> To use your own data, replace `data/transactions.csv` with a file
> containing `customer_id, order_date, category, quantity, order_value, discount_used`.

## Key outputs

- **`elbow_silhouette.png`** — objective justification for the chosen number of clusters
- **`pca_clusters.png`** — 2D PCA projection showing clean segment separation
- **`segment_profiles.csv`** — mean RFM values, customer count, and revenue share per segment, each auto-labeled with a business persona
- **`segment_revenue_contribution.png`** — bar chart of % revenue vs. % customers per segment (classic "80/20" view)
- **`cluster_vs_ground_truth.csv`** — validation cross-tab proving the unsupervised clusters recovered the true underlying customer types
- **`rfm_features.csv`** — the engineered feature table, ready for reuse in a dashboard tool

## Techniques demonstrated

- RFM (Recency, Frequency, Monetary) feature engineering + behavioral extensions
- Log-transformation for skewed distributions
- Feature scaling (StandardScaler)
- K-Means clustering with **data-driven** hyperparameter selection
- Dimensionality reduction (PCA) for visualization
- Silhouette-score model evaluation
- Cluster interpretation → business persona translation
- Revenue-contribution / Pareto-style segment analysis
