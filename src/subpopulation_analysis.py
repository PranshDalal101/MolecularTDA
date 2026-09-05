"""
subpopulation_analysis.py
===========================
Post-hoc statistical subpopulation analysis on Mapper nodes (manuscript
Section 3.4.4 / 4.3), run on the combined dataset's two Mapper graphs
(UMAP combined = Figure 12, t-SNE combined = Figure 11).

For each Mapper node, molecular (continuous) features and fingerprint
(binary) bits are scored separately, since they need different
treatments. Cohen's d is computed per node as (cluster_mean -
global_mean) / pooled_std(cluster, global_dataset), using the standard
pooled-variance formula, and molecular features are ranked by |d|.
Fingerprint bit importance is the signed difference between a node's
mean bit value and the bit's global frequency, ranked by magnitude. The
top 5 of each are kept per node.

Outputs
-------
  outputs/subpopulation_umap_combined.csv
  outputs/subpopulation_tsne_combined.csv
Each row is one Mapper node: id, sample count, dominant disease, purity,
plus its top-5 molecular features (name + Cohen's d) and top-5
fingerprint bits (bit + signed difference).
"""

import os

import numpy as np
import pandas as pd

from common_preprocessing import OUTPUTS_DIR, load_combined
from mapper_combined import build_tsne_combined_mapper, build_umap_combined_mapper

TOP_N = 5


def cohens_d(cluster_vals: np.ndarray, global_vals: np.ndarray) -> float:
    """Pooled-standard-deviation Cohen's d between a cluster's values and
    the full (global) dataset's values for one feature."""
    n1, n2 = len(cluster_vals), len(global_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = cluster_vals.var(ddof=1), global_vals.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((cluster_vals.mean() - global_vals.mean()) / pooled_std)


def load_manual_and_fingerprint_frames():
    """Splits the combined feature matrix (as built by
    common_preprocessing.load_combined) back into its manual/property
    columns and its fingerprint-bit columns, by construction consistent
    with how load_combined concatenates them."""
    df, X_combined = load_combined()
    manual_cols = df.select_dtypes(include=[np.number]).columns
    n_manual = len(manual_cols)

    manual_df = pd.DataFrame(X_combined[:, :n_manual], columns=manual_cols)
    fp_cols = [f"Bit_{i + 1}" for i in range(X_combined.shape[1] - n_manual)]
    fp_df = pd.DataFrame(X_combined[:, n_manual:], columns=fp_cols)
    return df, manual_df, fp_df


def analyze_node(members, manual_df, fp_df, global_manual, global_fp_freq, top_n=TOP_N):
    cluster_manual = manual_df.iloc[members]
    cluster_fp = fp_df.iloc[members]

    d_scores = {
        col: cohens_d(cluster_manual[col].values, global_manual[col].values)
        for col in manual_df.columns
    }
    top_features = sorted(d_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    bit_diffs = (cluster_fp.mean() - global_fp_freq).to_dict()
    top_bits = sorted(bit_diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    return top_features, top_bits


def run_subpopulation_analysis(graph, df, manual_df, fp_df, top_n=TOP_N) -> pd.DataFrame:
    global_manual = manual_df
    global_fp_freq = fp_df.mean()
    diseases = df["Disease"].astype(str).values

    rows = []
    for node_id, members in graph["nodes"].items():
        if not members:
            continue
        n = len(members)
        vc = pd.Series(diseases[members]).value_counts()
        dominant_disease, purity = vc.index[0], vc.iloc[0] / n

        top_features, top_bits = analyze_node(members, manual_df, fp_df, global_manual, global_fp_freq, top_n)

        row = {"node_id": node_id, "n_samples": n, "dominant_disease": dominant_disease,
               "purity": round(purity, 4)}
        for i, (name, d) in enumerate(top_features, 1):
            row[f"feature_{i}"] = name
            row[f"feature_{i}_cohens_d"] = round(d, 4)
        for i, (bit, diff) in enumerate(top_bits, 1):
            row[f"bit_{i}"] = bit
            row[f"bit_{i}_diff"] = round(diff, 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("purity", ascending=False).reset_index(drop=True)


def print_notable_clusters(table: pd.DataFrame, label: str, n=5, min_samples=4):
    print(f"\n=== {label}: top {n} clusters by purity (n_samples >= {min_samples}) ===")
    subset = table[table["n_samples"] >= min_samples].head(n)
    for _, row in subset.iterrows():
        feats = ", ".join(
            f"{row[f'feature_{i}']} (d={row[f'feature_{i}_cohens_d']})"
            for i in range(1, 4) if pd.notna(row.get(f"feature_{i}"))
        )
        bits = ", ".join(
            f"{row[f'bit_{i}']} (diff={row[f'bit_{i}_diff']})"
            for i in range(1, 3) if pd.notna(row.get(f"bit_{i}"))
        )
        print(f"  node {row['node_id']}: {row['dominant_disease']}, "
              f"n={row['n_samples']}, purity={row['purity']:.1%}")
        print(f"    top features: {feats}")
        print(f"    top bits: {bits}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df, manual_df, fp_df = load_manual_and_fingerprint_frames()

    print("Building UMAP combined mapper (Figure 12) ...")
    graph_umap, _, _, df_umap = build_umap_combined_mapper()
    table_umap = run_subpopulation_analysis(graph_umap, df_umap, manual_df, fp_df)
    umap_path = os.path.join(OUTPUTS_DIR, "subpopulation_umap_combined.csv")
    table_umap.to_csv(umap_path, index=False)
    print(f"Saved -> {umap_path} ({len(table_umap)} nodes)")

    print("\nBuilding t-SNE combined mapper (Figure 11) ...")
    graph_tsne, _, df_tsne = build_tsne_combined_mapper()
    table_tsne = run_subpopulation_analysis(graph_tsne, df_tsne, manual_df, fp_df)
    tsne_path = os.path.join(OUTPUTS_DIR, "subpopulation_tsne_combined.csv")
    table_tsne.to_csv(tsne_path, index=False)
    print(f"Saved -> {tsne_path} ({len(table_tsne)} nodes)")

    print_notable_clusters(table_umap, "UMAP combined")
    print_notable_clusters(table_tsne, "t-SNE combined")


if __name__ == "__main__":
    main()
