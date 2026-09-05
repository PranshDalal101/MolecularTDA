"""
persistent_homology.py
========================
Persistent homology analysis per disease (manuscript Section 3.6,
results in Section 4.5 / Figure 14 / Table 2).

For each disease, a Vietoris-Rips filtration (Euclidean distance) is
built on that disease's pharmacokinetic/physicochemical compounds up to
dimension 1 (H0 = connected components, H1 = loops), using `ripser` with
cocycles enabled. Betti curves count live H0/H1 features across a range
of filtration values; infinite death times are capped at the largest
finite death observed for that disease.

For each disease, the H1 interval with the longest lifetime is taken as
its most persistent loop. Its representative cocycle gives a binary
membership vector (which compounds participate in an edge of that
cycle), which is then Spearman-correlated against every molecular
feature to find what drives that loop's structure.

Outputs
-------
  figures/figure14_betti_curves_and_barcodes.png — panel A: H0/H1 Betti
    curves per disease; panel B: H0/H1 persistence barcodes per disease
  outputs/table2_persistent_h1_correlations.csv — significant (p<0.05)
    feature correlations with the most persistent H1 loop, per disease
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.stats import spearmanr

from common_preprocessing import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    get_pharmacokinetic_physicochemical_columns,
    load_pharmacokinetic_physicochemical,
)

P_VALUE_THRESHOLD = 0.05
N_EPS_STEPS = 200


def load_disease_datasets():
    """Splits the pharmacokinetic/physicochemical feature matrix by
    disease. Drugs used for more than one disease appear in more than one
    disease's dataset, matching the manuscript's "separating drugs from
    the full dataset" per disease."""
    df, X = load_pharmacokinetic_physicochemical()
    raw_cols = get_pharmacokinetic_physicochemical_columns(df)
    feature_names = list(df[raw_cols].select_dtypes(include=[np.number]).columns)

    datasets = {}
    for disease in sorted(df["Disease"].dropna().unique()):
        mask = (df["Disease"] == disease).values
        datasets[disease] = {
            "drug_names": df.loc[mask, "Drug Name"].values,
            "X": X[mask],
            "feature_names": feature_names,
        }
    return datasets


def compute_persistence(X, maxdim=1):
    """Vietoris-Rips filtration (Euclidean distance) up to H1, with
    cocycles enabled so representative cycles can be recovered."""
    return ripser(X, maxdim=maxdim, do_cocycles=True)


def betti_curve(dgm, eps_values, global_max_eps):
    """Betti number (count of live intervals) at each epsilon in
    eps_values. Infinite deaths are capped at global_max_eps."""
    if len(dgm) == 0:
        return np.zeros(len(eps_values))
    births = dgm[:, 0]
    deaths = np.where(np.isinf(dgm[:, 1]), global_max_eps, dgm[:, 1])
    return np.array([np.sum((births <= e) & (deaths > e)) for e in eps_values])


def most_persistent_h1(result, n_points):
    """Returns (membership, birth, death, persistence) for the H1
    interval with maximum persistence, or None if there are no H1
    features. membership is a 0/1 array over the n_points compounds,
    marking which participate in the representative cocycle's edges."""
    dgm1 = result["dgms"][1]
    if len(dgm1) == 0:
        return None

    finite = np.isfinite(dgm1[:, 1])
    if not finite.any():
        return None

    persistence = np.where(finite, dgm1[:, 1] - dgm1[:, 0], -np.inf)
    idx = int(np.argmax(persistence))
    birth, death = dgm1[idx]

    cocycle = result["cocycles"][1][idx]
    involved = set()
    for edge in cocycle:
        involved.add(int(edge[0]))
        involved.add(int(edge[1]))

    membership = np.zeros(n_points)
    membership[list(involved)] = 1
    return membership, birth, death, persistence[idx]


def correlate_features_with_loop(membership, X, feature_names):
    """Spearman rho/p-value between the H1 loop membership vector and
    every molecular feature, sorted by p-value ascending."""
    if membership.sum() == 0 or membership.sum() == len(membership):
        return []  # constant membership vector, no correlation possible

    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.std(col) == 0:
            continue
        rho, p = spearmanr(membership, col)
        if np.isnan(rho):
            continue
        results.append((name, rho, p))
    return sorted(results, key=lambda r: r[2])


def plot_betti_and_barcodes(all_results, out_path):
    """One figure combining Betti curves (row A, top) and persistence
    barcodes (row B, bottom) for every disease, matching manuscript
    Figure 14's two-panel layout."""
    diseases = list(all_results.keys())
    n = len(diseases)
    ncols = n

    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8), squeeze=False)

    for col, disease in enumerate(diseases):
        r = all_results[disease]

        ax = axes[0, col]
        ax.plot(r["eps_values"], r["betti_0"], label="H0", color="#1f77b4")
        ax.plot(r["eps_values"], r["betti_1"], label="H1", color="#d62728")
        ax.set_title(disease, fontsize=10)
        ax.set_xlabel("epsilon")
        if col == 0:
            ax.set_ylabel("Betti number")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        dgm0, dgm1 = r["dgm0"], r["dgm1"]
        y = 0
        for birth, death in dgm0:
            d = r["global_max_eps"] if np.isinf(death) else death
            ax.hlines(y, birth, d, color="#1f77b4", linewidth=1.5)
            y += 1
        y += 2
        for birth, death in dgm1:
            d = r["global_max_eps"] if np.isinf(death) else death
            ax.hlines(y, birth, d, color="#d62728", linewidth=1.5)
            y += 1
        ax.set_xlabel("epsilon")
        ax.set_yticks([])

    fig.text(0.01, 0.97, "A", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.48, "B", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=(0.02, 0, 1, 1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def run_persistent_homology_analysis():
    datasets = load_disease_datasets()
    all_results = {}
    table2_rows = []

    for disease, data in datasets.items():
        X, feature_names = data["X"], data["feature_names"]
        print(f"{disease}: {X.shape[0]} compounds, {X.shape[1]} features")

        result = compute_persistence(X, maxdim=1)
        dgm0, dgm1 = result["dgms"][0], result["dgms"][1]

        finite_deaths = np.concatenate([
            dgm0[np.isfinite(dgm0[:, 1]), 1] if len(dgm0) else np.array([]),
            dgm1[np.isfinite(dgm1[:, 1]), 1] if len(dgm1) else np.array([]),
        ])
        global_max_eps = float(finite_deaths.max()) if len(finite_deaths) else 1.0

        eps_values = np.linspace(0, global_max_eps, N_EPS_STEPS)
        betti_0 = betti_curve(dgm0, eps_values, global_max_eps)
        betti_1 = betti_curve(dgm1, eps_values, global_max_eps)

        all_results[disease] = {
            "eps_values": eps_values, "betti_0": betti_0, "betti_1": betti_1,
            "dgm0": dgm0, "dgm1": dgm1, "global_max_eps": global_max_eps,
        }

        loop = most_persistent_h1(result, X.shape[0])
        if loop is None:
            print(f"  no finite H1 loop found for {disease}")
            continue
        membership, birth, death, persistence = loop
        print(f"  most persistent H1 loop: birth={birth:.3f} death={death:.3f} persistence={persistence:.3f}")

        correlations = correlate_features_with_loop(membership, X, feature_names)
        significant = [c for c in correlations if c[2] < P_VALUE_THRESHOLD]
        for name, rho, p in significant:
            table2_rows.append({
                "Disease": disease, "Property": name,
                "Spearman_rho": round(rho, 4), "P_value": p,
            })

    plot_betti_and_barcodes(all_results, os.path.join(FIGURES_DIR, "figure14_betti_curves_and_barcodes.png"))

    table2 = pd.DataFrame(table2_rows)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUTS_DIR, "table2_persistent_h1_correlations.csv")
    table2.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv} ({len(table2)} significant feature correlations)")

    return all_results, table2


if __name__ == "__main__":
    run_persistent_homology_analysis()
