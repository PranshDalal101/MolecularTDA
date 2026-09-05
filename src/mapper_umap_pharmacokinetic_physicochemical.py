"""
mapper_umap_pharmacokinetic_physicochemical.py
================================================
UMAP mapper on pharmacokinetic and physicochemical properties (Table 1;
Figure 8). Lens: UMAP. Clusterer: DBSCAN. Parameters were found by the
optimization sweep in mapper_umap_pharmacokinetic_physicochemical_sweep.py.
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder

import kmapper as km
from common_preprocessing import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    load_pharmacokinetic_physicochemical,
    render_mapper_png,
)

OUTPUT_DIR = OUTPUTS_DIR


def _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path, title):
    mapper = km.KeplerMapper(verbose=1)

    tooltips = np.array([
        f"Drug: {name}<br>Disease: {disease}"
        for name, disease in zip(drug_names, y_labels)
    ])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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


def build_mapper(output_name="mapper_umap_pharmacokinetic_physicochemical_figure8.html",
                  png_out=os.path.join(FIGURES_DIR, "figure8_umap_pharmacokinetic_physicochemical.png")):
    df, X = load_pharmacokinetic_physicochemical()
    y_labels = df["Disease"].astype(str)
    drug_names = df.get("Drug Name", df.index)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    disease_mapping = {n: c for n, c in zip(le.classes_, range(len(le.classes_)))}

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=40,
        min_dist=0.15,
        metric="euclidean",
        init="spectral",
        spread=0.5,
        learning_rate=0.5,
        n_epochs=100,
        random_state=42,
        low_memory=False,
        verbose=False,
    )
    lens = reducer.fit_transform(X)

    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens, X,
        cover=km.Cover(n_cubes=19, perc_overlap=0.2),
        clusterer=DBSCAN(eps=38, min_samples=3),
    )

    if png_out:
        os.makedirs(os.path.dirname(png_out) or ".", exist_ok=True)
        render_mapper_png(graph, y_labels.values, png_out)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    _visualize(graph, y_encoded, disease_mapping, drug_names, y_labels, output_path,
              title="UMAP Physicochemical and Pharmacokinetic (Figure 8)")
    return graph


if __name__ == "__main__":
    build_mapper()
