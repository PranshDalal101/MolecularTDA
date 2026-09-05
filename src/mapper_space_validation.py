"""
mapper_space_validation.py
============================
Leave-one-out validation of the mapper space's drug-projection mechanism
(manuscript Section 3.5.2 / 4.4, Figure 13), standing in for testing on
genuinely new compounds like Paliperidone or RG8700.

Projecting a real new compound requires the raw-property mean/std used
when `data/zscore.csv` was originally standardized, which isn't available
in this repo (see the note in `mapper_space.py`). This script sidesteps
that gap: for a stratified sample of compounds already in the dataset, it
removes each one, refits the UMAP-combined mapper space (Figure 12's
parameters) on everyone else, and projects the held-out compound back in
using its own already-standardized row from `data/zscore.csv` — no raw-
to-z-score transform needed, since it was standardized once already, as
part of the whole dataset. This validates the same mechanism Figure 13
demonstrates (out-of-sample UMAP projection + nearest-node lookup), but
with a ground-truth disease label to check the result against instead of
qualitative judgment.

A "hit" is counted when the held-out compound's own disease matches the
majority disease of its nearest node in the refitted space.

Output: outputs/mapper_space_validation.csv, printed summary with hit rate.
"""

import os

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN

import kmapper as km
from common_preprocessing import OUTPUTS_DIR, load_combined
from mapper_space import DBSCAN_PARAMS, MAPPER_COVER_PARAMS, UMAP_PARAMS

N_PER_DISEASE = 4
RANDOM_SEED = 42


def stratified_holdout_sample(df, n_per_disease=N_PER_DISEASE, seed=RANDOM_SEED):
    """Picks up to n_per_disease compounds from each disease to hold out,
    so the validation covers every disease group rather than just the
    most common ones."""
    rng = np.random.RandomState(seed)
    holdout_idx = []
    for disease, group in df.groupby("Disease"):
        n = min(n_per_disease, len(group))
        holdout_idx.extend(rng.choice(group.index.values, size=n, replace=False))
    return sorted(holdout_idx)


def build_space_excluding(X_combined, exclude_idx):
    """Fits UMAP + Mapper (Figure 12's parameters) on every row except
    exclude_idx. Returns (reducer, graph, embedding, train_positions),
    where train_positions maps training-set row position -> original
    dataframe index."""
    train_positions = [i for i in range(X_combined.shape[0]) if i != exclude_idx]
    X_train = X_combined[train_positions]

    reducer = umap.UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(X_train)

    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        embedding, X_train,
        cover=km.Cover(**MAPPER_COVER_PARAMS),
        clusterer=DBSCAN(**DBSCAN_PARAMS),
    )
    return reducer, graph, embedding, train_positions


def nearest_node(graph, embedding, query_embedding):
    centroids = {
        node_id: embedding[members].mean(axis=0)
        for node_id, members in graph["nodes"].items() if members
    }
    if not centroids:
        return None
    return min(centroids, key=lambda nid: np.linalg.norm(centroids[nid] - query_embedding))


def run_validation(n_per_disease=N_PER_DISEASE):
    df, X_combined = load_combined()
    diseases = df["Disease"].astype(str).values
    drug_names = df.get("Drug Name", df.index)

    holdout_idx = stratified_holdout_sample(df, n_per_disease)
    print(f"Holding out {len(holdout_idx)} compounds across {df['Disease'].nunique()} diseases")

    rows = []
    for i, idx in enumerate(holdout_idx, 1):
        true_disease = diseases[idx]
        drug_name = drug_names.iloc[idx] if hasattr(drug_names, "iloc") else drug_names[idx]
        print(f"[{i}/{len(holdout_idx)}] holding out {drug_name} ({true_disease}) ...")

        reducer, graph, embedding, train_positions = build_space_excluding(X_combined, idx)

        query_vec = X_combined[idx].reshape(1, -1)
        query_embedding = reducer.transform(query_vec)[0]

        node_id = nearest_node(graph, embedding, query_embedding)
        if node_id is None:
            rows.append({"drug_name": drug_name, "true_disease": true_disease,
                        "nearest_node": None, "node_majority_disease": None,
                        "node_size": 0, "hit": False})
            continue

        members = graph["nodes"][node_id]
        member_orig_idx = [train_positions[m] for m in members]
        member_diseases = diseases[member_orig_idx]
        majority_disease = pd.Series(member_diseases).value_counts().idxmax()

        rows.append({
            "drug_name": drug_name,
            "true_disease": true_disease,
            "nearest_node": str(node_id),
            "node_majority_disease": majority_disease,
            "node_size": len(members),
            "hit": majority_disease == true_disease,
        })

    table = pd.DataFrame(rows)
    hit_rate = table["hit"].mean() if len(table) else 0.0

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "mapper_space_validation.csv")
    table.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"\nHit rate: {table['hit'].sum()}/{len(table)} ({hit_rate:.1%})")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run_validation()
