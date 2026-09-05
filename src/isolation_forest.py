"""
isolation_forest.py
====================
Isolation Forest anomaly detection on the combined (pharmacokinetic +
physicochemical + structural) drug-property dataset, visualized on both a
UMAP and a t-SNE embedding (manuscript Section 3.2.3 / Figure 6).

Model: IsolationForest(contamination=0.05, n_estimators=100), fit on the
combined feature matrix. UMAP and t-SNE embeddings of that same matrix
are used purely to visualize the anomaly scores in 2D — they don't need
to match the mapper figures' parameters, since they're just a projection
for plotting, not a Mapper graph.

Outputs: figures/figure6_isolation_forest.pdf / .png (2x2 grid: binary
anomaly classification + continuous anomaly score, for each embedding).
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

import os

from common_preprocessing import FIGURES_DIR, load_combined

# ── 1. Load & prep ───────────────────────────────────────────────────────
df_valid, X_combined = load_combined()

# ── 2. Isolation Forest (shared across both embeddings) ─────────────────
iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
labels = iso.fit_predict(X_combined)          # -1 anomaly, 1 normal
scores = -iso.decision_function(X_combined)   # higher = more anomalous
is_anomaly = labels == -1

# ── 3. UMAP embedding (for visualization only) ──────────────────────────
umap_model = umap.UMAP(
    n_components=3,
    n_neighbors=30,
    min_dist=0.3,
    metric="cosine",
    init="spectral",
    spread=1.0,
    learning_rate=1.0,
    random_state=42,
    verbose=False,
)
X_umap_3d = umap_model.fit_transform(X_combined)
X_umap = X_umap_3d[:, :2]

# ── 4. t-SNE embedding (for visualization only) ─────────────────────────
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
X_tsne_3d = tsne_model.fit_transform(X_combined)
X_tsne = X_tsne_3d[:, :2]

# ── 5. Figure (2x2 grid: UMAP/t-SNE x binary/continuous) ────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NORMAL_C = "#378ADD"
ANOMALY_C = "#D85A30"
S_SMALL = 12
S_GRAD = 16
ALPHA = 0.75
EW = 0.15

fig = plt.figure(figsize=(8, 7.5), dpi=300)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

for row, label in enumerate(["UMAP", "t-SNE"]):
    fig.text(0.5, 0.97 - row * 0.487, label, ha="center", va="top",
              fontsize=11, fontweight="bold", color="#222222")

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
embeddings = [X_umap, X_umap, X_tsne, X_tsne]
panel_labels = ["A", "B", "C", "D"]

for i, (ax, emb) in enumerate(zip(axes, embeddings)):
    binary = i % 2 == 0

    if binary:
        ax.scatter(emb[~is_anomaly, 0], emb[~is_anomaly, 1], c=NORMAL_C,
                   s=S_SMALL, alpha=ALPHA, linewidths=EW, edgecolors="k",
                   rasterized=True)
        ax.scatter(emb[is_anomaly, 0], emb[is_anomaly, 1], c=ANOMALY_C,
                   s=S_SMALL * 1.6, alpha=0.95, linewidths=EW,
                   edgecolors="k", rasterized=True, zorder=3)
        legend_els = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=NORMAL_C,
                   markersize=5, label="Normal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ANOMALY_C,
                   markersize=5, label="Anomaly"),
        ]
        ax.legend(handles=legend_els, frameon=False, fontsize=7.5,
                  loc="upper right", handletextpad=0.3)
        ax.set_title("Anomaly classification", fontsize=9, pad=4)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=scores, cmap="viridis",
                        s=S_GRAD, alpha=ALPHA, linewidths=EW,
                        edgecolors="k", rasterized=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Anomaly score", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.4)
        ax.set_title("Anomaly score (continuous)", fontsize=9, pad=4)

    ax.text(-0.08, 1.06, panel_labels[i], transform=ax.transAxes,
           fontsize=10, fontweight="bold", va="top")

    method = "UMAP" if i < 2 else "t-SNE"
    ax.set_xlabel(f"{method} 1", fontsize=8)
    ax.set_ylabel(f"{method} 2", fontsize=8)
    ax.tick_params(length=2)

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "figure6_isolation_forest.pdf"), dpi=300, bbox_inches="tight", backend="pdf")
plt.savefig(os.path.join(FIGURES_DIR, "figure6_isolation_forest.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved PDF + PNG ->", FIGURES_DIR)

# ── 6. Inspect anomalous rows ─────────────────────────────────────────
df_valid["anomaly_label"] = labels
df_valid["anomaly_score"] = scores
anomalies = df_valid[df_valid["anomaly_label"] == -1]
print(f"\nAnomalous rows: {len(anomalies)} / {len(df_valid)}")
print(anomalies.sort_values("anomaly_score", ascending=False).head(10))
