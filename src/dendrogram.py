"""
dendrogram.py
=============
Hierarchical clustering dendrogram of drug compounds by their combined
pharmacokinetic, physicochemical, and structural properties, colored by
disease association (manuscript Section 3.3.2, Figure 5).

Duplicate drug names are dropped (keeping the first occurrence), drugs
used across more than one disease are tracked separately and colored
black rather than by a single disease, and clustering uses Ward linkage
on the combined pharmacokinetic/physicochemical/structural feature
matrix.

Output: figures/figure5_dendrogram.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns

from common_preprocessing import (
    FIGURES_DIR,
    decode_fingerprint,
    load_numeric_properties,
    load_raw,
)

MIXED_DISEASE_COLOR = "black"
MIXED_DISEASE_LABEL = "Overlapping Drugs"


def load_deduped_compounds(path=None):
    """Load the compound table, drop duplicate drug names (keep first
    occurrence), and identify drugs associated with more than one
    disease before dedup collapses them to a single row."""
    df = load_raw(path) if path else load_raw()
    print("Loaded:", df.shape)

    mixed_drugs = set(
        df.groupby("Drug Name")["Disease"].nunique().loc[lambda s: s > 1].index
    )
    print(f"Drugs appearing in multiple diseases: {len(mixed_drugs)}")

    df = df.drop_duplicates(subset=["Drug Name"]).reset_index(drop=True)
    print("After duplicate removal:", df.shape)
    return df, mixed_drugs


def build_feature_matrix(df):
    """Combined pharmacokinetic/physicochemical + decoded fingerprint
    feature matrix, mean-imputed. Rows with an undecodable fingerprint
    are dropped."""
    decoded = df["Fingerprint2D"].apply(decode_fingerprint)
    valid = decoded.notnull()
    removed = (~valid).sum()
    if removed:
        print(f"  {removed} rows removed (fingerprint decode failed)")

    df = df[valid].reset_index(drop=True)
    fp_matrix = np.vstack(decoded[valid].values)

    manual_df = load_numeric_properties(df)
    combined = np.hstack([manual_df.values, fp_matrix])
    print(f"Feature matrix: {combined.shape}")
    return df, combined


def plot_dendrogram(df, X, mixed_drugs, out_path=None):
    out_path = out_path or os.path.join(FIGURES_DIR, "figure5_dendrogram.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    drug_labels = df["Drug Name"].astype(str).values
    disease_labels = df["Disease"].astype(str).values
    drug_to_disease = dict(zip(drug_labels, disease_labels))

    unique_diseases = np.unique(disease_labels)
    palette = sns.color_palette("hls", len(unique_diseases))
    disease_color_map = dict(zip(unique_diseases, palette))

    linked = sch.linkage(X, method="ward")

    # Scale figure width to the number of leaves so labels have enough
    # horizontal room not to overlap (fixed widths like 24in collapse
    # hundreds of drug names into an illegible smear).
    n_leaves = len(drug_labels)
    fig_width = max(24, n_leaves * 0.30)
    plt.figure(figsize=(fig_width, 14))
    dendro = sch.dendrogram(
        linked, labels=drug_labels, leaf_rotation=90, leaf_font_size=10,
        color_threshold=0, above_threshold_color="black",
    )

    ax = plt.gca()
    for lbl in ax.get_xmajorticklabels():
        drug_name = lbl.get_text().strip()
        if drug_name in mixed_drugs:
            lbl.set_color(MIXED_DISEASE_COLOR)
        else:
            lbl.set_color(disease_color_map.get(drug_to_disease.get(drug_name), "black"))

    for disease, color in disease_color_map.items():
        plt.plot([], [], marker="o", linestyle="", color=color, label=disease)
    plt.plot([], [], marker="o", linestyle="", color=MIXED_DISEASE_COLOR, label=MIXED_DISEASE_LABEL)
    plt.legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title("Hierarchical Clustering Dendrogram\n(Manual Features + Decoded Fingerprints)", fontsize=18)
    plt.ylabel("Distance", fontsize=14)
    plt.xlabel("Drug Name", fontsize=14)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()

    return dendro, drug_to_disease


def print_drug_order(dendro, drug_to_disease, mixed_drugs):
    ordered_drugs = dendro["ivl"]
    print("\n" + "=" * 60)
    print("DRUG ORDER IN DENDROGRAM")
    print("=" * 60)
    for i, drug in enumerate(ordered_drugs, start=1):
        disease = drug_to_disease.get(drug, "Unknown")
        overlap_tag = " [OVERLAPPING]" if drug in mixed_drugs else ""
        print(f"{i:03d}. {drug} (Disease: {disease}){overlap_tag}")
    print("=" * 60)
    print(f"Total drugs listed: {len(ordered_drugs)}")


def main():
    df, mixed_drugs = load_deduped_compounds()
    df, X = build_feature_matrix(df)
    dendro, drug_to_disease = plot_dendrogram(df, X, mixed_drugs)
    print_drug_order(dendro, drug_to_disease, mixed_drugs)


if __name__ == "__main__":
    main()
