"""
heatmap.py
==========
Spearman correlation heatmap between structural (fingerprint) and
pharmacokinetic/physicochemical properties (manuscript Section 3.3.1,
Figure 4).

Spearman correlations are computed between every fingerprint bit and
every physicochemical/pharmacokinetic property. Bits and properties are
each ranked by their mean absolute correlation across the other axis, and
the top 40 bits and top 20 properties are kept for the heatmap.

Bit substructure names (optional)
----------------------------------
If you have PubChem's official fingerprint key list as a PDF, drop it at
`data/list_fingerprints.pdf` and each bit will be labeled with its
substructure description. Without it (or without `pdfplumber` installed),
bits just fall back to generic "Bit_<n>" labels — the heatmap still
renders correctly either way.

Output: figures/figure4_correlation_heatmap.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_preprocessing import (
    DATA_DIR,
    FIGURES_DIR,
    decode_fingerprint,
    load_numeric_properties,
    load_raw,
)

N_SUBSTRUCTURE = 881
TOP_N_BITS = 40
TOP_N_PROPERTIES = 20  # readability cap for the property axis; not specified in the manuscript
FINGERPRINT_KEY_PDF = os.path.join(DATA_DIR, "list_fingerprints.pdf")


def load_pubchem_bit_names(pdf_path=FINGERPRINT_KEY_PDF):
    """Parse PubChem's fingerprint key PDF into {bit_position: substructure_name}.
    Returns an empty dict if the PDF (or pdfplumber) isn't available."""
    if not os.path.exists(pdf_path):
        return {}
    try:
        import pdfplumber
    except ImportError:
        print(f"  (pdfplumber not installed — bits will use generic labels)")
        return {}

    names = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    names[int(parts[0])] = " ".join(parts[1:])
    return names


def load_fingerprint_matrix(df):
    """Decode Fingerprint2D (881 real substructure bits) for every
    compound. Returns (df, fp_df) with df re-indexed to match."""
    decoded = df["Fingerprint2D"].apply(decode_fingerprint)
    valid = decoded.notnull()
    df = df[valid].reset_index(drop=True)
    decoded = decoded[valid].reset_index(drop=True)

    sample_len = len(decoded.iloc[0])
    assert sample_len == N_SUBSTRUCTURE, f"Expected {N_SUBSTRUCTURE} bits, got {sample_len}"
    print(f"Decoded {len(decoded)} drugs x {sample_len} substructure bits")

    bit_names = load_pubchem_bit_names()
    columns = [
        f"Bit {i + 1}: {bit_names[i + 1]}" if (i + 1) in bit_names else f"Bit_{i + 1}"
        for i in range(N_SUBSTRUCTURE)
    ]
    fp_df = pd.DataFrame(np.vstack(decoded.values), columns=columns)

    fp_df = fp_df.loc[:, fp_df.var() > 0]  # drop zero-variance bits (uninformative)
    print(f"Bits remaining after variance filter: {fp_df.shape[1]}")
    return df, fp_df


def load_property_matrix(df, exclude=("CID", "Fingerprint2D")):
    """All numeric pharmacokinetic/physicochemical columns, mean-imputed,
    with zero-variance columns dropped (uninformative for correlation)."""
    prop_df = load_numeric_properties(df, exclude=exclude)
    prop_df = prop_df.loc[:, prop_df.var() > 0]
    print(f"Properties after variance filter: {prop_df.shape[1]}")
    return prop_df


def compute_top_correlations(fp_df, prop_df, top_bits=TOP_N_BITS, top_properties=TOP_N_PROPERTIES):
    """Spearman correlation between every bit and every property, then
    keep the bits and properties with the highest *mean* absolute
    correlation (per manuscript Section 3.3.1: "arithmetically averaged
    ... ranked based on their mean correlations")."""
    combined = pd.concat([fp_df, prop_df], axis=1)
    full_corr = combined.corr(method="spearman")
    cross_corr = full_corr.loc[fp_df.columns, prop_df.columns].fillna(0.0)

    top_bits_idx = cross_corr.abs().mean(axis=1).nlargest(top_bits).index
    cross_zoom = cross_corr.loc[top_bits_idx, :]

    top_props_idx = cross_zoom.abs().mean(axis=0).nlargest(top_properties).index
    return cross_zoom[top_props_idx]


def plot_heatmap(cross_zoom, out_path=None):
    out_path = out_path or os.path.join(FIGURES_DIR, "figure4_correlation_heatmap.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig_h = max(12, len(cross_zoom) * 0.35)
    fig_w = max(10, len(cross_zoom.columns) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cross_zoom, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, square=True,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.3, linecolor="grey",
        cbar_kws={"label": "Spearman rho", "shrink": 0.6},
    )
    ax.set_xlabel("Pharmacokinetic and Physicochemical Properties", fontsize=11)
    ax.set_ylabel("Structural Properties (PubChem Substructure Bits)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()


def main():
    df = load_raw()
    df, fp_df = load_fingerprint_matrix(df)

    min_len = min(len(df), len(fp_df))
    df, fp_df = df.iloc[:min_len].reset_index(drop=True), fp_df.iloc[:min_len].reset_index(drop=True)

    prop_df = load_property_matrix(df)

    cross_zoom = compute_top_correlations(fp_df, prop_df)
    plot_heatmap(cross_zoom)


if __name__ == "__main__":
    main()
