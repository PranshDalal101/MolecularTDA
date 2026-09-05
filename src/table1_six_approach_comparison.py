"""
table1_six_approach_comparison.py
===================================
Regenerates manuscript Table 1 (Section 4.2.1, "Six-Approach Comparison
Table") by running each of the six final mapper configurations and
computing nodes, edges, mean purity, connectivity, and edge ratio for
each — using the same metric definitions as the sweep scripts
(`common_preprocessing.compute_graph_metrics`).

Run this after any change to the shared preprocessing (e.g. the
fingerprint-decoding fix) to see how the six approaches' numbers actually
compare today, against what's printed in the manuscript.

Output: outputs/table1_six_approach_comparison.csv, printed to stdout.
"""

import os

import pandas as pd

from common_preprocessing import (
    OUTPUTS_DIR,
    compute_graph_metrics,
    load_pharmacokinetic_physicochemical,
    load_structural_fingerprints,
)
from mapper_combined import build_tsne_combined_mapper, build_umap_combined_mapper
from mapper_tsne_pharmacokinetic_physicochemical import build_mapper as tsne_pharm_phys
from mapper_tsne_structural_fingerprint import build_mapper as tsne_structural
from mapper_umap_pharmacokinetic_physicochemical import build_mapper as umap_pharm_phys
from mapper_umap_structural_fingerprint import build_mapper as umap_structural

# Manuscript Table 1, for side-by-side comparison.
MANUSCRIPT_TABLE1 = {
    "Umap combined":                          dict(n_edges=407, n_nodes=138, mean_purity=0.3668, connectivity=0.043055, edge_ratio=2.9493),
    "Umap pharmacokinetic and physicochemical": dict(n_edges=88, n_nodes=49, mean_purity=0.3092, connectivity=0.9592, edge_ratio=1.7959),
    "Umap fingerprints":                       dict(n_edges=37, n_nodes=22, mean_purity=0.286, connectivity=0.9545, edge_ratio=1.6818),
    "Tsne combined":                           dict(n_edges=55, n_nodes=35, mean_purity=0.3349, connectivity=0.9714, edge_ratio=1.5714),
    "t-SNE pharmacokinetic and physicochemical": dict(n_edges=81, n_nodes=49, mean_purity=0.3334, connectivity=1.0, edge_ratio=1.6531),
    "t-SNE Structural":                        dict(n_edges=84, n_nodes=40, mean_purity=0.3192, connectivity=1.0, edge_ratio=2.1),
}

ROW_ORDER = list(MANUSCRIPT_TABLE1.keys())


def current_metrics():
    rows = {}

    print("Building UMAP combined + t-SNE combined ...")
    graph_umap_c, _, _, df_c = build_umap_combined_mapper()
    rows["Umap combined"] = compute_graph_metrics(graph_umap_c, df_c["Disease"].astype(str).values)

    graph_tsne_c, _, df_c2 = build_tsne_combined_mapper()
    rows["Tsne combined"] = compute_graph_metrics(graph_tsne_c, df_c2["Disease"].astype(str).values)

    print("Building UMAP pharmacokinetic/physicochemical (Figure 8) ...")
    graph_umap_pp = umap_pharm_phys()
    df_pp, _ = load_pharmacokinetic_physicochemical()
    rows["Umap pharmacokinetic and physicochemical"] = compute_graph_metrics(
        graph_umap_pp, df_pp["Disease"].astype(str).values
    )

    print("Building UMAP structural/fingerprint (Figure 10, rank 2) ...")
    graph_umap_fp = umap_structural()
    df_fp, _ = load_structural_fingerprints()
    rows["Umap fingerprints"] = compute_graph_metrics(graph_umap_fp, df_fp["Disease"].astype(str).values)

    print("Building t-SNE pharmacokinetic/physicochemical (Figure 7) ...")
    graph_tsne_pp = tsne_pharm_phys()
    rows["t-SNE pharmacokinetic and physicochemical"] = compute_graph_metrics(
        graph_tsne_pp, df_pp["Disease"].astype(str).values
    )

    print("Building t-SNE structural (Figure 9) ...")
    graph_tsne_fp = tsne_structural()
    rows["t-SNE Structural"] = compute_graph_metrics(graph_tsne_fp, df_fp["Disease"].astype(str).values)

    return rows


def build_comparison_table(rows):
    records = []
    for approach in ROW_ORDER:
        current = rows[approach]
        manuscript = MANUSCRIPT_TABLE1[approach]
        records.append({
            "Approach": approach,
            "edges (current)": current["n_edges"],
            "edges (manuscript)": manuscript["n_edges"],
            "nodes (current)": current["n_nodes"],
            "nodes (manuscript)": manuscript["n_nodes"],
            "mean purity (current)": current["mean_purity"],
            "mean purity (manuscript)": manuscript["mean_purity"],
            "connectivity (current)": current["connectivity"],
            "connectivity (manuscript)": manuscript["connectivity"],
            "edge ratio (current)": current["edge_ratio"],
            "edge ratio (manuscript)": manuscript["edge_ratio"],
        })
    return pd.DataFrame.from_records(records)


def main():
    rows = current_metrics()
    table = build_comparison_table(rows)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "table1_six_approach_comparison.csv")
    table.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
