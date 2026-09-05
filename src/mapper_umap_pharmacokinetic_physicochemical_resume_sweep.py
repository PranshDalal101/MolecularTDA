"""
mapper_umap_pharmacokinetic_physicochemical_resume_sweep.py
=============================================================
Companion to mapper_umap_pharmacokinetic_physicochemical_sweep.py.

The original Phase-1 random search crashed partway through. Rather than
re-running Phase 1 from scratch, this script skips straight to Phase 2 and
zooms around the 5 best combinations that had already been found before
the crash (hardcoded in KNOWN_WINNERS below, taken from the last saved
console output / live log). Uses the same scoring function, UMAP/DBSCAN
run logic, and zoom-grid builder as the main sweep script.
"""

import itertools
import os
import time
import warnings

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN

from common_preprocessing import LOGS_DIR, load_pharmacokinetic_physicochemical
from mapper_umap_pharmacokinetic_physicochemical_sweep import (
    TARGET_NODE_SIZE, MIN_NODES, MAX_NODES,
    W_STRUCTURE, W_PURITY, W_EDGE_DENSITY,
    score_graph, zoom_range,
)

warnings.filterwarnings("ignore")

LIVE_LOG_CSV = os.path.join(LOGS_DIR, "umap_basic_RESUME_live_log.csv")
PHASE2_CSV = os.path.join(LOGS_DIR, "umap_basic_RESUME_phase2_leaderboard.csv")

PHASE2_TIME_BUDGET = int(2.75 * 3600)
SECONDS_PER_COMBO = 3

# Top winners from the crashed run — hardcoded from its last console output.
KNOWN_WINNERS = [
    {"score": 0.7797, "n_components": 2, "n_neighbors": 50, "min_dist": 0.25,
     "metric": "euclidean", "init": "spectral", "spread": 1.0, "learning_rate": 1.0,
     "n_epochs": 200, "n_cubes": 15, "perc_overlap": 0.3, "eps": 41, "min_samples": 5,
     "n_nodes": 67, "mean_purity": 0.3362},
    {"score": 0.7741, "n_components": 2, "n_neighbors": 75, "min_dist": 0.25,
     "metric": "euclidean", "init": "spectral", "spread": 1.0, "learning_rate": 1.0,
     "n_epochs": 200, "n_cubes": 15, "perc_overlap": 0.3, "eps": 41, "min_samples": 5,
     "n_nodes": 75, "mean_purity": 0.3511},
    {"score": 0.7694, "n_components": 2, "n_neighbors": 50, "min_dist": 0.5,
     "metric": "euclidean", "init": "spectral", "spread": 1.0, "learning_rate": 1.0,
     "n_epochs": 200, "n_cubes": 15, "perc_overlap": 0.3, "eps": 41, "min_samples": 5,
     "n_nodes": 60, "mean_purity": 0.316},
    {"score": 0.7666, "n_components": 2, "n_neighbors": 100, "min_dist": 0.1,
     "metric": "euclidean", "init": "spectral", "spread": 1.0, "learning_rate": 1.0,
     "n_epochs": 200, "n_cubes": 15, "perc_overlap": 0.3, "eps": 41, "min_samples": 5,
     "n_nodes": 23, "mean_purity": 0.2827},
    {"score": 0.7635, "n_components": 2, "n_neighbors": 20, "min_dist": 0.5,
     "metric": "euclidean", "init": "spectral", "spread": 1.0, "learning_rate": 1.0,
     "n_epochs": 200, "n_cubes": 15, "perc_overlap": 0.3, "eps": 21, "min_samples": 5,
     "n_nodes": 70, "mean_purity": 0.328},
]


def run_combo(X, diseases, umap_p, mapper_p):
    try:
        reducer = umap.UMAP(
            n_components=umap_p["n_components"], n_neighbors=umap_p["n_neighbors"],
            min_dist=umap_p["min_dist"], metric=umap_p["metric"], spread=umap_p["spread"],
            learning_rate=umap_p["learning_rate"], n_epochs=umap_p["n_epochs"],
            init=umap_p["init"], random_state=42, low_memory=False, verbose=False,
        )
        lens = reducer.fit_transform(X)
        mapper = km.KeplerMapper(verbose=0)
        graph = mapper.map(
            lens, X,
            cover=km.Cover(n_cubes=mapper_p["n_cubes"], perc_overlap=mapper_p["perc_overlap"]),
            clusterer=DBSCAN(eps=mapper_p["eps"], min_samples=mapper_p["min_samples"]),
        )
        score, meta = score_graph(graph, diseases)
        return score, {**umap_p, **mapper_p, **meta}
    except Exception as e:
        return -1.0, {**umap_p, **mapper_p, "score": -1.0, "n_nodes": 0, "n_edges": 0,
                      "connectivity": 0.0, "median_size": 0.0, "mean_purity": 0.0,
                      "edge_ratio": 0.0, "error": str(e)[:120]}


def append_row(row_dict, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.exists(filepath)
    pd.DataFrame([row_dict]).to_csv(filepath, mode="a", header=write_header, index=False)


def build_zoom_grids(winner):
    zoom_umap = {
        "n_components": [int(winner["n_components"])],
        "metric": [winner["metric"]],
        "init": [winner["init"]],
        "n_neighbors": zoom_range(winner["n_neighbors"], 5, lo=2, hi=150, as_int=True),
        "min_dist": zoom_range(winner["min_dist"], 0.05, lo=0.0, hi=0.99),
        "spread": zoom_range(winner["spread"], 0.25, lo=0.1, hi=3.0),
        "learning_rate": zoom_range(winner["learning_rate"], 0.25, lo=0.1, hi=3.0),
        "n_epochs": zoom_range(winner["n_epochs"], 100, lo=100, hi=2000, as_int=True),
        "random_state": [42], "low_memory": [False], "verbose": [False],
    }
    zoom_mapper = {
        "n_cubes": zoom_range(winner["n_cubes"], 2, lo=3, hi=50, as_int=True),
        "perc_overlap": zoom_range(winner["perc_overlap"], 0.05, lo=0.05, hi=0.75),
        "eps": zoom_range(winner["eps"], 3, lo=1, hi=50, as_int=True),
        "min_samples": zoom_range(winner["min_samples"], 1, lo=2, hi=20, as_int=True),
    }
    return zoom_umap, zoom_mapper


def resume_zoom(X, diseases, time_budget):
    all_results = []
    best_score = max(w["score"] for w in KNOWN_WINNERS)
    phase_start = time.time()

    for rank, winner in enumerate(KNOWN_WINNERS, 1):
        if time.time() - phase_start >= time_budget:
            break
        zoom_umap, zoom_mapper = build_zoom_grids(winner)
        tk, mk = list(zoom_umap.keys()), list(zoom_mapper.keys())
        combos = [
            (dict(zip(tk, t)), dict(zip(mk, m)))
            for t in itertools.product(*[zoom_umap[k] for k in tk])
            for m in itertools.product(*[zoom_mapper[k] for k in mk])
        ]
        budget_left = time_budget - (time.time() - phase_start)
        combos = combos[:int(budget_left / SECONDS_PER_COMBO)]

        for j, (umap_p, mapper_p) in enumerate(combos, 1):
            if time.time() - phase_start >= time_budget:
                break
            t0 = time.time()
            score, meta = run_combo(X, diseases, umap_p, mapper_p)
            new_best = score > best_score
            if new_best:
                best_score = score
            meta.update(phase=f"zoom{rank}", zoom_rank=rank, combo_index=j,
                        elapsed_s=round(time.time() - t0, 2), new_best=new_best,
                        best_so_far=round(best_score, 6))
            append_row(meta, LIVE_LOG_CSV)
            all_results.append(meta)
            print(f"    [{j}/{len(combos)}] score={score:.4f} best={best_score:.4f}")

    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results).sort_values("score", ascending=False).reset_index(drop=True)


def main():
    df, X = load_pharmacokinetic_physicochemical()
    diseases = df["Disease"].astype(str).reset_index(drop=True)

    phase2_df = resume_zoom(X, diseases, PHASE2_TIME_BUDGET)
    if phase2_df.empty:
        print("No results — check your data path")
        return
    phase2_df.to_csv(PHASE2_CSV, index=False)
    best = phase2_df.iloc[0]
    print(f"\nBest: score={best['score']:.6f} nodes={int(best['n_nodes'])} "
          f"purity={best['mean_purity']:.4f}")


if __name__ == "__main__":
    main()
