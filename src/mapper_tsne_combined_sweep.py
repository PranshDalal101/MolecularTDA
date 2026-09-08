"""
mapper_tsne_combined_sweep.py
===============================
Parameter optimization sweep for the t-SNE combined mapper (Table 1
"Tsne combined"; Figure 11): random search for 2 hours, then an
exhaustive zoom around the top 5 winners for 45 minutes, run on the
combined (pharmacokinetic + physicochemical + structural) feature matrix.
This is the search that produced the parameters used in
mapper_combined.py:build_tsne_combined_mapper().
"""

import itertools
import os
import random
import time
import warnings

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
from kmapper import Cover
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from common_preprocessing import LOGS_DIR, load_combined

warnings.filterwarnings("ignore")

LIVE_LOG_CSV = os.path.join(LOGS_DIR, "tsne_combined_live_log.csv")
PHASE1_CSV = os.path.join(LOGS_DIR, "tsne_combined_phase1_leaderboard.csv")
PHASE2_CSV = os.path.join(LOGS_DIR, "tsne_combined_phase2_leaderboard.csv")

PHASE1_TIME_BUDGET = 2 * 3600
PHASE2_TIME_BUDGET = 45 * 60
SECONDS_PER_COMBO = 3

TARGET_NODE_SIZE = 20
MIN_NODES = 5
MAX_NODES = 200
W_STRUCTURE = 0.50
W_PURITY = 0.30
W_EDGE_DENSITY = 0.20
TOP_K_WINNERS = 5

TSNE_SPACE = {
    "n_components": [2, 3],
    "perplexity": [5, 10, 15, 20, 30, 40, 50, 75, 100],
    "early_exaggeration": [8, 12, 24],
    "learning_rate": ["auto", 100, 200, 500],
    "metric": ["cosine", "euclidean"],
    "init": ["pca", "random"],
    "max_iter": [1000], "n_iter_without_progress": [300], "min_grad_norm": [1e-7],
    "method": ["barnes_hut"], "angle": [0.5], "random_state": [42],
    "metric_params": [None], "n_jobs": [None], "verbose": [0],
}

MAPPER_SPACE = {
    "n_cubes": [5, 10, 15, 20, 25],
    "perc_overlap": [0.1, 0.2, 0.3, 0.4, 0.5],
    "eps": list(range(1, 51, 10)),
    "min_samples": [3, 5, 10],
}


def sample_combo(space):
    return {k: random.choice(v) for k, v in space.items()}


def score_graph(graph, diseases):
    nodes = graph["nodes"]
    n_nodes = len(nodes)
    if n_nodes == 0:
        return 0.0, {"n_nodes": 0, "n_edges": 0, "connectivity": 0.0,
                     "median_size": 0.0, "mean_purity": 0.0,
                     "edge_ratio": 0.0, "score": 0.0}

    G = km.adapter.to_nx(graph)
    n_edges = G.number_of_edges()

    largest_cc = max(nx.connected_components(G), key=len) if n_nodes > 1 else set(nodes.keys())
    connectivity = len(largest_cc) / n_nodes

    sizes = [len(v) for v in nodes.values()]
    median_size = np.median(sizes)
    size_score = np.exp(-((median_size - TARGET_NODE_SIZE) ** 2) / (2 * TARGET_NODE_SIZE ** 2))
    count_score = 1.0 if MIN_NODES <= n_nodes <= MAX_NODES else 0.3
    structure = 0.5 * connectivity + 0.3 * size_score + 0.2 * count_score

    purities = []
    for samples in nodes.values():
        if not samples:
            continue
        vc = diseases.iloc[samples].value_counts()
        purities.append(vc.iloc[0] / vc.sum())
    mean_purity = np.mean(purities) if purities else 0.0

    edge_ratio = n_edges / n_nodes
    edge_score = np.exp(-((edge_ratio - 1.5) ** 2) / (2 * 1.5 ** 2))

    score = W_STRUCTURE * structure + W_PURITY * mean_purity + W_EDGE_DENSITY * edge_score

    return score, {
        "n_nodes": n_nodes, "n_edges": n_edges,
        "connectivity": round(connectivity, 4), "median_size": round(median_size, 2),
        "mean_purity": round(mean_purity, 4), "edge_ratio": round(edge_ratio, 4),
        "score": round(score, 6),
    }


def run_combo(X, diseases, tsne_p, mapper_p):
    try:
        lens = TSNE(
            n_components=tsne_p["n_components"], perplexity=tsne_p["perplexity"],
            early_exaggeration=tsne_p["early_exaggeration"], learning_rate=tsne_p["learning_rate"],
            max_iter=tsne_p["max_iter"], n_iter_without_progress=tsne_p["n_iter_without_progress"],
            min_grad_norm=tsne_p["min_grad_norm"], metric=tsne_p["metric"],
            metric_params=tsne_p["metric_params"], init=tsne_p["init"],
            verbose=tsne_p["verbose"], random_state=tsne_p["random_state"],
            method=tsne_p["method"], angle=tsne_p["angle"], n_jobs=tsne_p["n_jobs"],
        ).fit_transform(X)

        mapper = km.KeplerMapper(verbose=0)
        graph = mapper.map(
            lens, X,
            cover=Cover(n_cubes=mapper_p["n_cubes"], perc_overlap=mapper_p["perc_overlap"]),
            clusterer=DBSCAN(eps=mapper_p["eps"], min_samples=mapper_p["min_samples"]),
        )
        score, meta = score_graph(graph, diseases)
        return score, {**tsne_p, **mapper_p, **meta}
    except Exception as e:
        return -1.0, {**tsne_p, **mapper_p, "score": -1.0, "n_nodes": 0, "n_edges": 0,
                      "connectivity": 0.0, "median_size": 0.0, "mean_purity": 0.0,
                      "edge_ratio": 0.0, "error": str(e)[:120]}


def append_row(row_dict, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.exists(filepath)
    pd.DataFrame([row_dict]).to_csv(filepath, mode="a", header=write_header, index=False)


def zoom_range(val, step, n_steps=2, lo=None, hi=None, as_int=False):
    vals = [val + step * i for i in range(-n_steps, n_steps + 1)]
    if lo is not None:
        vals = [v for v in vals if v >= lo]
    if hi is not None:
        vals = [v for v in vals if v <= hi]
    if as_int:
        return sorted(set(int(round(v)) for v in vals)) or [int(val)]
    return sorted(set(round(v, 5) for v in vals)) or [round(val, 5)]


def build_zoom_grids(winner):
    lr = winner["learning_rate"]
    zoom_tsne = {
        "n_components": [int(winner["n_components"])],
        "metric": [winner["metric"]],
        "init": [winner["init"]],
        "perplexity": zoom_range(winner["perplexity"], 5, lo=2, hi=150, as_int=True),
        "early_exaggeration": zoom_range(winner["early_exaggeration"], 4, lo=4, hi=48, as_int=True),
        "learning_rate": [lr] if isinstance(lr, str) else zoom_range(lr, 50, lo=10, hi=1000, as_int=True),
        "max_iter": [1000], "n_iter_without_progress": [300], "min_grad_norm": [1e-7],
        "method": ["barnes_hut"], "angle": [0.5], "random_state": [42],
        "metric_params": [None], "n_jobs": [None], "verbose": [0],
    }
    zoom_mapper = {
        "n_cubes": zoom_range(winner["n_cubes"], 3, lo=3, hi=50, as_int=True),
        "perc_overlap": zoom_range(winner["perc_overlap"], 0.05, lo=0.05, hi=0.75),
        "eps": zoom_range(winner["eps"], 5, lo=1, hi=50, as_int=True),
        "min_samples": zoom_range(winner["min_samples"], 1, lo=2, hi=20, as_int=True),
    }
    return zoom_tsne, zoom_mapper


def phase1_random(X, diseases, time_budget):
    best_score = -1.0
    results = []
    seen = set()
    phase_start = time.time()
    i = 0
    while True:
        if time.time() - phase_start >= time_budget:
            break
        tsne_p = sample_combo(TSNE_SPACE)
        mapper_p = sample_combo(MAPPER_SPACE)
        key = (str(sorted(tsne_p.items())), str(sorted(mapper_p.items())))
        if key in seen:
            continue
        seen.add(key)
        i += 1
        t0 = time.time()
        score, meta = run_combo(X, diseases, tsne_p, mapper_p)
        new_best = score > best_score
        if new_best:
            best_score = score
        meta.update(phase="phase1", combo_index=i, elapsed_s=round(time.time() - t0, 2),
                    new_best=new_best, best_so_far=round(best_score, 6))
        append_row(meta, LIVE_LOG_CSV)
        results.append(meta)
        print(f"  [{i}] score={score:.4f} best={best_score:.4f} nodes={meta.get('n_nodes','?')}")
    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


def phase2_zoom(X, diseases, phase1_df, time_budget):
    all_results = []
    best_score = -1.0
    phase_start = time.time()
    for rank, (_, winner) in enumerate(phase1_df.head(TOP_K_WINNERS).iterrows(), 1):
        if time.time() - phase_start >= time_budget:
            break
        zoom_tsne, zoom_mapper = build_zoom_grids(winner)
        tk, mk = list(zoom_tsne.keys()), list(zoom_mapper.keys())
        combos = [
            (dict(zip(tk, t)), dict(zip(mk, m)))
            for t in itertools.product(*[zoom_tsne[k] for k in tk])
            for m in itertools.product(*[zoom_mapper[k] for k in mk])
        ]
        budget_left = time_budget - (time.time() - phase_start)
        combos = combos[:int(budget_left / SECONDS_PER_COMBO)]
        for j, (tsne_p, mapper_p) in enumerate(combos, 1):
            if time.time() - phase_start >= time_budget:
                break
            t0 = time.time()
            score, meta = run_combo(X, diseases, tsne_p, mapper_p)
            new_best = score > best_score
            if new_best:
                best_score = score
            meta.update(phase=f"phase2_zoom{rank}", zoom_rank=rank, combo_index=j,
                        elapsed_s=round(time.time() - t0, 2), new_best=new_best,
                        best_so_far=round(best_score, 6))
            append_row(meta, LIVE_LOG_CSV)
            all_results.append(meta)
            print(f"    [{j}/{len(combos)}] score={score:.4f} best={best_score:.4f}")
    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results).sort_values("score", ascending=False).reset_index(drop=True)


def run_parameter_sweep():
    for f in [LIVE_LOG_CSV, PHASE1_CSV, PHASE2_CSV]:
        if os.path.exists(f):
            os.remove(f)
    random.seed(99)

    df, X = load_combined()
    diseases = df["Disease"].astype(str).reset_index(drop=True)

    phase1_df = phase1_random(X, diseases, PHASE1_TIME_BUDGET)
    phase1_df.to_csv(PHASE1_CSV, index=False)

    phase2_df = phase2_zoom(X, diseases, phase1_df, PHASE2_TIME_BUDGET)
    if not phase2_df.empty:
        phase2_df.to_csv(PHASE2_CSV, index=False)
        best = phase2_df.iloc[0]
    else:
        best = phase1_df.iloc[0]

    print(f"\nBest: score={best['score']:.6f} nodes={int(best['n_nodes'])} "
          f"purity={best['mean_purity']:.4f}")
    return best


if __name__ == "__main__":
    run_parameter_sweep()
