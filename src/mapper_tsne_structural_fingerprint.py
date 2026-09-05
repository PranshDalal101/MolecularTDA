"""
mapper_tsne_structural_fingerprint.py
=======================================
t-SNE mapper on decoded structural fingerprints (Table 1 "t-SNE
Structural"; Figure 9). Lens: t-SNE. Clusterer: DBSCAN. Parameters were
found by the optimization sweep in
mapper_tsne_structural_fingerprint_sweep.py.
"""

import os

import numpy as np
from kmapper import Cover
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

import kmapper as km
from common_preprocessing import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    load_structural_fingerprints,
    render_mapper_png,
)


def build_mapper(output_html=os.path.join(OUTPUTS_DIR, "tsne_structural_figure9.html"),
                  png_out=os.path.join(FIGURES_DIR, "figure9_tsne_structural_fingerprint.png")):
    df, X = load_structural_fingerprints()
    labels = df["Disease"].astype(str)

    lens = TSNE(
        n_components=2,
        perplexity=5,
        early_exaggeration=8,
        learning_rate="auto",
        metric="euclidean",
        init="pca",
        max_iter=1000,
        random_state=42,
    ).fit_transform(X)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X,
        cover=Cover(n_cubes=7, perc_overlap=0.15),
        clusterer=DBSCAN(eps=21, min_samples=6),
    )

    if png_out:
        render_mapper_png(graph, labels.values, png_out)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    tooltips = np.array([f"Disease: {d}" for d in labels])

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    mapper.visualize(
        graph, color_values=y_encoded, color_function_name="Disease",
        path_html=output_html, title="t-SNE Structural (Figure 9)",
        custom_tooltips=tooltips,
    )
    print(f"Saved -> {output_html}  (nodes={len(graph['nodes'])})")
    return graph


if __name__ == "__main__":
    build_mapper()
