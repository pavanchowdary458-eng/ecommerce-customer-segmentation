"""
01_generate_data.py
--------------------
Generates a synthetic but realistic e-commerce transactions dataset,
built from underlying customer "archetypes" (VIP, loyal, discount-seeker,
one-time, churn-risk, new) so that the clustering step later has genuine
latent structure to recover — the mark of a well-designed segmentation
exercise rather than clustering pure noise.

Output:
    data/transactions.csv
    data/customers.csv   (ground-truth archetype kept ONLY for validation,
                           not used as a model feature)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(7)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_CUSTOMERS = 2500
TODAY = datetime(2026, 9, 1)
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
              "Books", "Groceries", "Toys", "Automotive", "Furniture"]

# Archetype definitions: (weight, order_freq_range, aov_range, tenure_days_range, recency_bias)
ARCHETYPES = {
    "VIP":            dict(weight=0.07, n_orders=(25, 60),  aov=(3000, 9000), tenure=(400, 1000), recency_days=(1, 20)),
    "Loyal":          dict(weight=0.18, n_orders=(12, 30),  aov=(1200, 3500), tenure=(250, 900),  recency_days=(1, 40)),
    "Discount-Seeker":dict(weight=0.20, n_orders=(8, 20),   aov=(300, 900),   tenure=(100, 700),  recency_days=(5, 60)),
    "New-Customer":   dict(weight=0.15, n_orders=(1, 3),    aov=(500, 2000),  tenure=(1, 60),     recency_days=(1, 30)),
    "Churn-Risk":     dict(weight=0.22, n_orders=(3, 10),   aov=(500, 2500),  tenure=(200, 800),  recency_days=(150, 400)),
    "One-Time":       dict(weight=0.18, n_orders=(1, 1),    aov=(200, 1500),  tenure=(1, 900),    recency_days=(30, 500)),
}

names = list(ARCHETYPES.keys())
weights = np.array([ARCHETYPES[n]["weight"] for n in names])
weights = weights / weights.sum()
assigned = RNG.choice(names, size=N_CUSTOMERS, p=weights)

customer_rows = []
txn_rows = []
customer_id_counter = 100000

for archetype in assigned:
    customer_id_counter += 1
    cid = f"CUST{customer_id_counter}"
    spec = ARCHETYPES[archetype]

    n_orders = RNG.integers(spec["n_orders"][0], spec["n_orders"][1] + 1)
    tenure_days = RNG.integers(spec["tenure"][0], spec["tenure"][1] + 1)
    signup_date = TODAY - timedelta(days=int(tenure_days))
    most_recent_gap = RNG.integers(spec["recency_days"][0], spec["recency_days"][1] + 1)
    last_order_date = TODAY - timedelta(days=int(most_recent_gap))

    # spread order dates between signup and last_order_date
    if n_orders > 1 and last_order_date > signup_date:
        span = (last_order_date - signup_date).days
        offsets = np.sort(RNG.integers(0, max(span, 1), size=n_orders - 1))
        order_dates = [signup_date + timedelta(days=int(o)) for o in offsets] + [last_order_date]
    else:
        order_dates = [last_order_date]

    preferred_categories = RNG.choice(CATEGORIES, size=RNG.integers(1, 4), replace=False)

    total_spend = 0
    for od in order_dates:
        aov = RNG.uniform(spec["aov"][0], spec["aov"][1])
        aov *= RNG.normal(1, 0.15)  # per-order noise
        aov = max(aov, 50)
        category = RNG.choice(preferred_categories)
        discount_used = 1 if archetype == "Discount-Seeker" and RNG.random() < 0.7 else (1 if RNG.random() < 0.15 else 0)
        quantity = RNG.integers(1, 6)
        total_spend += aov
        txn_rows.append({
            "customer_id": cid,
            "order_date": od.date().isoformat(),
            "category": category,
            "quantity": int(quantity),
            "order_value": round(aov, 2),
            "discount_used": discount_used,
        })

    customer_rows.append({
        "customer_id": cid,
        "signup_date": signup_date.date().isoformat(),
        "true_archetype": archetype,  # ground truth, held out from modeling
    })

customers = pd.DataFrame(customer_rows)
transactions = pd.DataFrame(txn_rows)

# inject light data-quality issues, as a real e-commerce export would have
dup_idx = RNG.choice(transactions.index, size=int(len(transactions) * 0.01), replace=False)
transactions = pd.concat([transactions, transactions.loc[dup_idx]], ignore_index=True)
null_idx = RNG.choice(transactions.index, size=int(len(transactions) * 0.005), replace=False)
transactions.loc[null_idx, "order_value"] = np.nan

transactions.to_csv(DATA_DIR / "transactions.csv", index=False)
customers.to_csv(DATA_DIR / "customers.csv", index=False)

print(f"Generated {len(customers)} customers and {len(transactions)} transactions")
print(customers["true_archetype"].value_counts())
