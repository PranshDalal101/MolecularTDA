"""
mapper_tsne_pharmacokinetic_physicochemical.py
================================================
t-SNE mapper on pharmacokinetic and physicochemical properties (Table 1;
Figure 7). Lens: t-SNE. Clusterer: DBSCAN.
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

import kmapper as km
from common_preprocessing import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    load_pharmacokinetic_physicochemical,
    render_mapper_png,
)

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "mapper_tsne_results")


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


def build_mapper(output_name="mapper_tsne_pharmacokinetic_physicochemical_figure7.html",
                  png_out=os.path.join(FIGURES_DIR, "figure7_tsne_pharmacokinetic_physicochemical.png")):
    df, X = load_pharmacokinetic_physicochemical()
    y_labels = df["Disease"].astype(str)
    drug_names = df.get("Drug Name", pd.Series(["N/A"] * len(df), index=df.index))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    disease_mapping = {n: c for n, c in zip(le.classes_, range(len(le.classes_)))}

    tsne_model = TSNE(
        n_components=2,
        perplexity=5,
        early_exaggeration=12,
        learning_rate=500,
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        init="pca",
        method="barnes_hut",
        angle=0.5,
        random_state=42,
    )
    lens = tsne_model.fit_transform(X)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X,
        cover=km.Cover(n_cubes=10, perc_overlap=0.2),
        clusterer=DBSCAN(eps=31, min_samples=5),
    )

    if png_out:
        os.makedirs(os.path.dirname(png_out) or ".", exist_ok=True)
        render_mapper_png(graph, y_labels.values, png_out)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path,
              title="t-SNE Physicochemical and Pharmacokinetic (Figure 7)")
    return graph


if __name__ == "__main__":
    build_mapper()
