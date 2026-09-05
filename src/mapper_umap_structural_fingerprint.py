"""
mapper_umap_structural_fingerprint.py
=======================================
UMAP mapper on decoded structural fingerprints (Table 1 "Umap
fingerprints"; Figure 10). Lens: UMAP. Clusterer: DBSCAN.
"""

import os

import umap
from kmapper import Cover
from sklearn.cluster import DBSCAN

import kmapper as km
from common_preprocessing import FIGURES_DIR, load_structural_fingerprints, render_mapper_png


def build_mapper(out_path=os.path.join(FIGURES_DIR, "figure10_umap_structural_fingerprint.png")):
    df, X = load_structural_fingerprints()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.5,
        metric="cosine",
        init="random",
        spread=0.5,
        learning_rate=0.5,
        random_state=42,
        verbose=False,
    )
    lens = reducer.fit_transform(X)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X,
        cover=Cover(n_cubes=10, perc_overlap=0.15),
        clusterer=DBSCAN(eps=41, min_samples=5),
    )

    render_mapper_png(graph, df["Disease"].astype(str).values, out_path)
    return graph


if __name__ == "__main__":
    build_mapper()
