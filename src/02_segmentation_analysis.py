"""
02_segmentation_analysis.py
------------------------------
Advanced customer segmentation pipeline:

  1. Data cleaning (dedupe, impute missing order values)
  2. RFM + behavioral feature engineering (Recency, Frequency, Monetary,
     AOV, category diversity, discount-sensitivity, tenure)
  3. Log-transform + scaling to handle skewed monetary/frequency features
  4. Optimal cluster count selection via Elbow method AND Silhouette score
  5. K-Means clustering
  6. PCA for 2D visualization of high-dimensional segments
  7. Business-facing segment profiling (named personas + revenue contribution)
  8. Cross-tab validation against the (held-out) ground-truth archetype
     used only to generate the data — demonstrating the clusters recovered
     genuine latent structure, not noise.

Outputs (outputs/):
    elbow_silhouette.png
    pca_clusters.png
    segment_profiles.csv
    segment_revenue_contribution.png
    cluster_vs_ground_truth.csv
    rfm_features.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pathlib import Path

sns.set_theme(style="whitegrid", palette="deep")
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TODAY = pd.Timestamp("2026-09-01")

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
txn = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["order_date"])
customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])

before = len(txn)
txn = txn.drop_duplicates()
txn["order_value"] = txn.groupby("customer_id")["order_value"].transform(lambda s: s.fillna(s.median()))
txn["order_value"] = txn["order_value"].fillna(txn["order_value"].median())
print(f"Cleaned transactions: {before} -> {len(txn)} rows (removed dupes, imputed nulls)")

# ---------------------------------------------------------------------------
# 2. RFM + BEHAVIORAL FEATURE ENGINEERING
# ---------------------------------------------------------------------------
agg = txn.groupby("customer_id").agg(
    recency_days=("order_date", lambda s: (TODAY - s.max()).days),
    frequency=("order_date", "count"),
    monetary=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    category_diversity=("category", "nunique"),
    discount_rate=("discount_used", "mean"),
    total_units=("quantity", "sum"),
).reset_index()

rfm = agg.merge(customers[["customer_id", "signup_date"]], on="customer_id")
rfm["tenure_days"] = (TODAY - rfm["signup_date"]).dt.days
rfm.drop(columns=["signup_date"], inplace=True)
rfm.to_csv(OUT_DIR / "rfm_features.csv", index=False)

FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value",
            "category_diversity", "discount_rate", "tenure_days"]

X = rfm[FEATURES].copy()
# log-transform heavily skewed monetary/frequency features (add 1 to handle zeros)
for col in ["frequency", "monetary", "avg_order_value", "recency_days", "tenure_days"]:
    X[col] = np.log1p(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 3. OPTIMAL K SELECTION: ELBOW + SILHOUETTE
# ---------------------------------------------------------------------------
k_range = range(2, 10)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#1F4E79")
axes[0].set_title("Elbow Method", weight="bold")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#C0504D")
axes[1].set_title("Silhouette Score by k", weight="bold")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average silhouette score")
best_k = list(k_range)[int(np.argmax(sil_scores))]
axes[1].axvline(best_k, color="gray", linestyle="--", label=f"Best k={best_k}")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "elbow_silhouette.png", dpi=150)
plt.close(fig)
print(f"Optimal k selected via silhouette score: k={best_k}")

# ---------------------------------------------------------------------------
# 4. FINAL K-MEANS FIT
# ---------------------------------------------------------------------------
final_k = best_k
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, rfm["cluster"])
print(f"Final model: k={final_k}, silhouette score={final_sil:.3f}")

# ---------------------------------------------------------------------------
# 5. PCA VISUALIZATION (2D projection of the 7-D feature space)
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
rfm["pca1"], rfm["pca2"] = coords[:, 0], coords[:, 1]
explained = pca.explained_variance_ratio_.sum()

fig, ax = plt.subplots(figsize=(9, 7))
palette = sns.color_palette("tab10", final_k)
for c in sorted(rfm["cluster"].unique()):
    sub = rfm[rfm["cluster"] == c]
    ax.scatter(sub["pca1"], sub["pca2"], s=25, alpha=0.6, color=palette[c], label=f"Cluster {c}")
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=250, c="black", label="Centroids", edgecolor="white", linewidth=1.5)
ax.set_title(f"Customer Segments — PCA Projection ({explained*100:.1f}% variance explained)", weight="bold")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "pca_clusters.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. BUSINESS-FACING SEGMENT PROFILES (auto-labeled personas)
# ---------------------------------------------------------------------------
profile = rfm.groupby("cluster")[FEATURES].mean().round(1)
profile["customer_count"] = rfm.groupby("cluster").size()
profile["pct_of_customers"] = (100 * profile["customer_count"] / len(rfm)).round(1)
profile["total_revenue"] = rfm.groupby("cluster")["monetary"].sum().round(0)
profile["pct_of_revenue"] = (100 * profile["total_revenue"] / rfm["monetary"].sum()).round(1)


def name_segment(row):
    if row["monetary"] > rfm["monetary"].quantile(0.75) and row["recency_days"] < rfm["recency_days"].median():
        return "Champions / VIP"
    if row["recency_days"] > rfm["recency_days"].quantile(0.75):
        return "At-Risk / Churning"
    if row["frequency"] <= 2:
        return "New / One-Time Buyers"
    if row["discount_rate"] > 0.4:
        return "Discount-Driven"
    return "Loyal / Core Customers"


profile["persona"] = profile.apply(name_segment, axis=1)
profile = profile.sort_values("pct_of_revenue", ascending=False)
profile.to_csv(OUT_DIR / "segment_profiles.csv")
print("\nSegment profiles:\n", profile[["persona", "customer_count", "pct_of_customers", "pct_of_revenue"]])

# Revenue contribution chart
fig, ax = plt.subplots(figsize=(9, 5.5))
plot_df = profile.reset_index().sort_values("pct_of_revenue")
bars = ax.barh(plot_df["persona"] + " (C" + plot_df["cluster"].astype(str) + ")",
                plot_df["pct_of_revenue"], color=sns.color_palette("viridis", len(plot_df)))
for bar, pct_cust in zip(bars, plot_df["pct_of_customers"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}% revenue | {pct_cust:.1f}% customers",
            va="center", fontsize=9)
ax.set_xlabel("% of Total Revenue")
ax.set_title("Revenue Contribution by Customer Segment", weight="bold")
ax.set_xlim(0, plot_df["pct_of_revenue"].max() * 1.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "segment_revenue_contribution.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. VALIDATION AGAINST GROUND-TRUTH ARCHETYPE (sanity check only)
# ---------------------------------------------------------------------------
validation = rfm.merge(customers[["customer_id", "true_archetype"]], on="customer_id")
crosstab = pd.crosstab(validation["true_archetype"], validation["cluster"])
crosstab.to_csv(OUT_DIR / "cluster_vs_ground_truth.csv")
print("\nCluster vs. ground-truth archetype cross-tab (validation only, not used in modeling):")
print(crosstab)

print(f"\nAll outputs written to: {OUT_DIR}")
