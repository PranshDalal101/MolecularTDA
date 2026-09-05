"""
mapper_combined.py
===================
Mapper on the combined (pharmacokinetic + physicochemical + structural)
dataset, for both lenses: UMAP (Table 1 "Umap combined"; Figure 12) and
t-SNE (Table 1 "Tsne combined"; Figure 11). The UMAP version is also the
basis for the fixed chemical space in mapper_space.py, since UMAP
supports projecting new points and t-SNE does not.
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

import kmapper as km
from common_preprocessing import FIGURES_DIR, OUTPUTS_DIR, load_combined, render_mapper_png

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "mapper_combined_results")


def _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapper = km.KeplerMapper(verbose=1)

    tooltips = np.array([
        f"Drug: {name}<br>Disease: {disease}"
        for name, disease in zip(drug_names, y_labels)
    ])

    mapper.visualize(graph, color_values=y_encoded, color_function_name="Disease",
                     path_html=output_path, title=title, custom_tooltips=tooltips)

    norm = mcolors.Normalize(vmin=y_encoded.min(), vmax=y_encoded.max())
    legend_html = "<h3>Disease Legend</h3><ul>"
    for disease, code in disease_mapping.items():
        color = mcolors.to_hex(cm.rainbow(norm(code)))
        legend_html += f"<li><span style='color:{color};'>●</span> {disease}: {code}</li>"
    legend_html += "</ul>"

    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", legend_html + "</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Saved -> {output_path}  (nodes={len(graph['nodes'])})")


def build_umap_combined_mapper(output_name="mapper_umap_combined_figure12.html",
                                png_out=os.path.join(FIGURES_DIR, "figure12_umap_combined.png")):
    """Figure 12: UMAP on the combined (pharmacokinetic + physicochemical
    + structural) dataset. Also the mapper used as the basis for the
    fixed chemical space in mapper_space.py."""
    df, X_combined = load_combined()
    y_labels = df["Disease"].astype(str)
    drug_names = df.get("Drug Name", df.index)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    disease_mapping = {n: c for n, c in zip(le.classes_, range(len(le.classes_)))}

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.5,
        metric="cosine",
        init="random",
        spread=0.5,
        learning_rate=0.5,
        random_state=42,
        verbose=False,
    )
    lens = reducer.fit_transform(X_combined)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X_combined,
        cover=km.Cover(n_cubes=10, perc_overlap=0.15),
        clusterer=DBSCAN(eps=41, min_samples=4),
    )

    if png_out:
        os.makedirs(os.path.dirname(png_out) or ".", exist_ok=True)
        render_mapper_png(graph, y_labels.values, png_out)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path,
              title="UMAP Combined (Figure 12)")
    return graph, reducer, X_combined, df


def build_tsne_combined_mapper(output_name="mapper_tsne_combined_figure11.html",
                                png_out=os.path.join(FIGURES_DIR, "figure11_tsne_combined.png")):
    """Figure 11: t-SNE on the combined (pharmacokinetic + physicochemical
    + structural) dataset. (t-SNE has no out-of-sample .transform(), so
    unlike the UMAP version this cannot be reused for the mapper space —
    see Section 3.5 of the manuscript.)"""
    df, X_combined = load_combined()
    y_labels = df["Disease"].astype(str)
    drug_names = df.get("Drug Name", df.index)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    disease_mapping = {n: c for n, c in zip(le.classes_, range(len(le.classes_)))}

    tsne_model = TSNE(
        n_components=3,
        perplexity=5,
        early_exaggeration=8,
        learning_rate="auto",
        metric="euclidean",
        init="random",
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        method="barnes_hut",
        angle=0.5,
        random_state=42,
        verbose=0,
    )
    lens = tsne_model.fit_transform(X_combined)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X_combined,
        cover=km.Cover(n_cubes=5, perc_overlap=0.1),
        clusterer=DBSCAN(eps=21, min_samples=12),
    )

    if png_out:
        os.makedirs(os.path.dirname(png_out) or ".", exist_ok=True)
        render_mapper_png(graph, y_labels.values, png_out)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path,
              title="t-SNE Combined (Figure 11)")
    return graph, X_combined, df


if __name__ == "__main__":
    build_umap_combined_mapper()
    build_tsne_combined_mapper()
