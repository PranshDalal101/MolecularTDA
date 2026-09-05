"""
common_preprocessing.py
========================
Shared data-loading and feature-matrix helpers used by every script in
this folder: the isolation forest, the six mappers, and the mapper space.

The master compound table is `data/zscore.csv`. Three feature sets are
used across the six mapper approaches (Table 1):
  1. "pharmacokinetic_physicochemical" — the numeric column block from
     'ic50' through 'ConformerCount3D' (pharmacokinetic + physicochemical
     + PubChem-computed descriptor columns). No fingerprint bits.
  2. "structural" — the Fingerprint2D column, base64-decoded into an
     881-bit binary vector (one bit per PubChem substructure key).
  3. "combined" — the two above, concatenated.

All paths below are resolved relative to this file's location, so every
script in `src/` works regardless of the caller's current directory.
"""

import base64
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import kmapper as km

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "zscore.csv")


def decode_fingerprint(fp):
    """Base64 PubChem Fingerprint2D string -> the 881 real substructure
    bits (0/1 float), per the CACTVS fingerprint layout: a 32-bit length
    header, 881 substructure bits (PubChem-indexed 1-881), then 7 padding
    bits (920 bits / 115 bytes total). Returns None on failure or if the
    decoded fingerprint is shorter than expected."""
    HEADER_BITS = 32
    N_SUBSTRUCTURE = 881
    if pd.isna(fp):
        return None
    try:
        raw = base64.b64decode(str(fp).strip())
        all_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        if len(all_bits) < HEADER_BITS + N_SUBSTRUCTURE:
            return None
        return all_bits[HEADER_BITS:HEADER_BITS + N_SUBSTRUCTURE].astype(np.float64)
    except Exception:
        return None


def load_raw(path=DEFAULT_DATA_PATH):
    """Load the master compound table."""
    return pd.read_csv(path)


def get_pharmacokinetic_physicochemical_columns(df):
    """The numeric block used by the manuscript's 'pharmacokinetic and
    physicochemical' dataset type: everything from 'ic50' to
    'ConformerCount3D' inclusive. Confirmed via the CSV header to contain
    no Fingerprint2D / bit columns."""
    start = df.columns.get_loc("ic50")
    end = df.columns.get_loc("ConformerCount3D")
    return df.columns[start:end + 1]


def load_pharmacokinetic_physicochemical(path=DEFAULT_DATA_PATH, dropna_disease=True):
    """Returns (df, X) where X is the raw (unscaled) numeric
    pharmacokinetic/physicochemical feature matrix, mean-imputed."""
    df = load_raw(path)
    if dropna_disease:
        df = df.dropna(subset=["Disease"]).reset_index(drop=True)

    feature_cols = get_pharmacokinetic_physicochemical_columns(df)
    X = df[feature_cols].select_dtypes(include=np.number)
    X = X.fillna(X.mean())
    return df, X.values


def load_numeric_properties(df, exclude=("Fingerprint2D",)):
    """All numeric columns in df except `exclude`, mean-imputed. Used
    where "manual"/"property" features means every numeric column in the
    table rather than just the ic50-ConformerCount3D block (e.g. the
    dendrogram and correlation heatmap, which both include every
    available numeric descriptor)."""
    numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    prop_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="mean").fit_transform(prop_df)
    return pd.DataFrame(imputed, columns=prop_df.columns)


def load_structural_fingerprints(path=DEFAULT_DATA_PATH):
    """Decode Fingerprint2D for every compound. Rows with an undecodable
    fingerprint are dropped. Returns (df, X) with df re-indexed to match X."""
    df = load_raw(path)

    decoded = df["Fingerprint2D"].apply(decode_fingerprint)
    valid = decoded.notnull()
    removed = (~valid).sum()

    df = df[valid].reset_index(drop=True)
    decoded = decoded[valid].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid fingerprints decoded from Fingerprint2D.")

    X = np.vstack(decoded.values)
    if removed:
        print(f"  ⚠ {removed} rows removed (fingerprint decode failed)")
    print(f"  ✓ {len(df)} molecules | fingerprint dim = {X.shape[1]}")
    return df, X


def load_combined(path=DEFAULT_DATA_PATH):
    """Pharmacokinetic/physicochemical columns + decoded fingerprint bits,
    concatenated (mean-imputed numeric block). Matches the 'combined'
    dataset used for the Isolation Forest and the UMAP/t-SNE combined
    mappers (Figures 11-12)."""
    df = load_raw(path)

    decoded = df["Fingerprint2D"].apply(decode_fingerprint)
    valid = decoded.notnull()
    df = df[valid].reset_index(drop=True)
    fp_matrix = np.vstack(decoded[valid].values)

    num_cols = df.select_dtypes(include=[np.number]).columns
    num_imputed = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

    X_combined = np.hstack([num_imputed, fp_matrix])
    print(f"Feature matrix: {X_combined.shape}")
    return df, X_combined


def compute_graph_metrics(graph, diseases):
    """Nodes, edges, mean disease purity, connectivity, and edge ratio for
    a Kepler Mapper graph — the columns of manuscript Table 1 (Six-Approach
    Comparison Table). `diseases` must be indexed the same way as the data
    passed into `mapper.map()` (i.e. aligned with `graph["nodes"]` member
    indices). Connectivity = fraction of nodes in the largest connected
    component; mean purity = average, across nodes, of each node's most
    common disease's share of that node's members."""
    nodes = graph["nodes"]
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {"n_nodes": 0, "n_edges": 0, "mean_purity": 0.0, "connectivity": 0.0, "edge_ratio": 0.0}

    G = km.adapter.to_nx(graph)
    n_edges = G.number_of_edges()

    largest_cc = max(nx.connected_components(G), key=len) if n_nodes > 1 else set(nodes.keys())
    connectivity = len(largest_cc) / n_nodes

    diseases = np.asarray(diseases)
    purities = []
    for members in nodes.values():
        if not members:
            continue
        vc = pd.Series(diseases[members]).value_counts()
        purities.append(vc.iloc[0] / vc.sum())
    mean_purity = float(np.mean(purities)) if purities else 0.0

    edge_ratio = n_edges / n_nodes

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_purity": round(mean_purity, 4),
        "connectivity": round(connectivity, 4),
        "edge_ratio": round(edge_ratio, 4),
    }


def render_mapper_png(graph, diseases, out_path, seed=42, figsize=(10, 8)):
    """Static, title-less PNG of a Kepler Mapper graph: networkx spring
    layout, nodes sized by member count and colored by majority disease,
    with a disease legend. Used as the canonical figure renderer for all
    six mapper scripts and the combined-dataset mappers."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    G = km.adapter.to_nx(graph)
    pos = nx.spring_layout(G, seed=seed)

    unique_diseases = sorted(pd.Series(diseases).astype(str).unique())
    cmap_colours = mpl.colormaps.get("tab20")
    disease_to_color = {d: cmap_colours(i % 20) for i, d in enumerate(unique_diseases)}

    node_ids = list(G.nodes())
    node_sizes, node_colors = [], []
    for nid in node_ids:
        members = graph["nodes"][nid]
        node_sizes.append(40 + 6 * len(members))
        if not members:
            node_colors.append("#cccccc")
        else:
            majority = pd.Series(np.array(diseases)[members]).astype(str).value_counts().idxmax()
            node_colors.append(disease_to_color.get(majority, "#cccccc"))

    n_legend_rows = -(-len(unique_diseases) // 3)  # ceil div, ncol=3
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 0.35 * n_legend_rows))
    if node_ids:
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                alpha=0.95, linewidths=0.5, edgecolors="black", ax=ax)
    if G.edges():
        nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.8, ax=ax)
    ax.axis("off")
    patches = [mpatches.Patch(color=disease_to_color[d], label=d) for d in unique_diseases]
    ax.legend(handles=patches, title="Disease", loc="lower left",
              bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, fontsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    n_edges = sum(len(v) for v in graph["links"].values())
    print(f"  -> {out_path}  (nodes={len(graph['nodes'])}, edges={n_edges})")
