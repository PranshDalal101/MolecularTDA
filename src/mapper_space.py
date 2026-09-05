"""
mapper_space.py
================
Fixed chemical space and drug projection pipeline (Section 3.5.2, Figure
3; results in Section 4.4). Builds the UMAP combined mapper (Figure 12)
as a reusable reference space and projects new compounds into it.

Why UMAP and not t-SNE: t-SNE has no out-of-sample `.transform()` —
projecting a new point requires recomputing the whole embedding, which
would shift every existing drug's coordinates. UMAP's `.transform()`
embeds a new point into an already-fitted space without moving anything
else, so it's the only one of the six approaches usable as a fixed
reference space for new compounds.

Dependency: `data/zscore.csv` contains drugs already standardized using
the mean/std of the original (pre-standardization) property table. To
project a genuinely new compound, apply those same per-column mean/std
values (and the log-transform used for IC50/EC50/KI) to its raw
properties before calling `project_query()` — recomputing mean/std from
`zscore.csv` itself would be wrong, since those columns are already
~N(0, 1). This script expects `project_query()` to be handed an
already-standardized feature vector (same column order as
`get_pharmacokinetic_physicochemical_columns()`) plus the raw
Fingerprint2D base64 string. If the normalization pipeline can export its
fitted per-column mean/std, wire it in at `STANDARDIZATION_PARAMS_PATH`
and `standardize_raw_row()` below instead of building the vector by hand.

Usage
-----
    space = MapperSpace.build()          # fit once, ~minutes
    space.save()                          # persist to outputs/mapper_space.pkl
    space = MapperSpace.load()

    result = space.project_query(
        drug_name="Paliperidone",
        standardized_features=my_standardized_vector,   # see note above
        fingerprint_base64=my_fingerprint2d_string,
    )
    print(result.summary())
"""

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN

import kmapper as km
from common_preprocessing import (
    DATA_DIR,
    OUTPUTS_DIR,
    decode_fingerprint,
    get_pharmacokinetic_physicochemical_columns,
    load_combined,
)

# Path to a JSON/CSV of {column: {"mean": ..., "std": ...}} produced by your
# existing raw-property z-score normalization script, if you want
# standardize_raw_row() to work end-to-end. Optional.
STANDARDIZATION_PARAMS_PATH = os.path.join(DATA_DIR, "standardization_params.json")

# Final UMAP-combined parameters (Figure 12 / Section 4.2.2), re-optimized
# after the fingerprint-decoding fix — see mapper_combined.py's docstring.
UMAP_PARAMS = dict(
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
MAPPER_COVER_PARAMS = dict(n_cubes=10, perc_overlap=0.15)
DBSCAN_PARAMS = dict(eps=41, min_samples=4)


def standardize_raw_row(raw_row: dict) -> np.ndarray:
    """Apply saved per-column mean/std (and the IC50/EC50/KI log-transform
    described in Section 3.2) to a dict of RAW property values, returning
    a standardized vector in the same column order as the training data.

    Requires STANDARDIZATION_PARAMS_PATH to exist — see the module
    docstring. Raises if it hasn't been produced yet.
    """
    import json
    with open(STANDARDIZATION_PARAMS_PATH) as f:
        params = json.load(f)

    df, _ = load_combined()
    columns = get_pharmacokinetic_physicochemical_columns(df)

    vec = np.zeros(len(columns))
    for i, col in enumerate(columns):
        val = raw_row.get(col, np.nan)
        if col in ("ic50", "ec50", "ki") and val is not None and val > 0:
            val = np.log10(val)
        mean = params[col]["mean"]
        std = params[col]["std"] or 1.0
        vec[i] = 0.0 if pd.isna(val) else (val - mean) / std
    return vec


@dataclass
class ProjectionResult:
    drug_name: str
    coordinates: np.ndarray
    nearest_node_id: str
    nearest_node_members: list
    nearest_node_diseases: pd.Series
    majority_disease: str

    def summary(self) -> str:
        coord_str = ", ".join(f"{c:.4f}" for c in self.coordinates)
        lines = [
            f"{self.drug_name} mapped at ({coord_str})",
            f"Nearest node: {self.nearest_node_id}",
            f"Members ({len(self.nearest_node_members)}): {', '.join(self.nearest_node_members)}",
            f"Most common disease in node: {self.majority_disease}",
        ]
        return "\n".join(lines)


@dataclass
class MapperSpace:
    """The fixed UMAP-combined chemical space (Figure 12) plus the fitted
    Mapper graph, reusable for projecting new query compounds without
    retraining (Section 3.5)."""

    reducer: umap.UMAP
    graph: dict
    embedding: np.ndarray
    df: pd.DataFrame
    feature_dim: int
    node_centroids: dict = field(default_factory=dict)

    @classmethod
    def build(cls, path=None):
        df, X_combined = load_combined(path) if path else load_combined()
        reducer = umap.UMAP(**UMAP_PARAMS)
        embedding = reducer.fit_transform(X_combined)

        mapper = km.KeplerMapper(verbose=1)
        graph = mapper.map(
            embedding, X_combined,
            cover=km.Cover(**MAPPER_COVER_PARAMS),
            clusterer=DBSCAN(**DBSCAN_PARAMS),
        )

        space = cls(reducer=reducer, graph=graph, embedding=embedding, df=df,
                    feature_dim=X_combined.shape[1])
        space._compute_node_centroids()
        return space

    def _compute_node_centroids(self):
        self.node_centroids = {
            node_id: self.embedding[members].mean(axis=0)
            for node_id, members in self.graph["nodes"].items()
            if members
        }

    def save(self, path=None):
        path = path or os.path.join(OUTPUTS_DIR, "mapper_space.pkl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved mapper space -> {path}")

    @staticmethod
    def load(path=None) -> "MapperSpace":
        path = path or os.path.join(OUTPUTS_DIR, "mapper_space.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def project_query(
        self,
        drug_name: str,
        standardized_features: np.ndarray,
        fingerprint_base64: str,
    ) -> ProjectionResult:
        """Embed a new compound into the fixed chemical space and report
        its nearest Mapper node (Figure 3, Figure 13)."""
        fp_bits = decode_fingerprint(fingerprint_base64)
        if fp_bits is None:
            raise ValueError(f"Could not decode fingerprint for {drug_name!r}.")
        query_vec = np.hstack([standardized_features, fp_bits]).reshape(1, -1)

        if query_vec.shape[1] != self.feature_dim:
            raise ValueError(
                f"Query feature vector has {query_vec.shape[1]} dims, "
                f"expected {self.feature_dim}. Check that standardized_features "
                f"matches get_pharmacokinetic_physicochemical_columns() order "
                f"and that the fingerprint decoded to the expected bit length."
            )

        query_embedding = self.reducer.transform(query_vec)[0]

        nearest_node_id = min(
            self.node_centroids,
            key=lambda nid: np.linalg.norm(self.node_centroids[nid] - query_embedding),
        )
        members = self.graph["nodes"][nearest_node_id]
        member_names = self.df.iloc[members]["Drug Name"].tolist()
        member_diseases = self.df.iloc[members]["Disease"]
        majority_disease = member_diseases.value_counts().idxmax()

        return ProjectionResult(
            drug_name=drug_name,
            coordinates=query_embedding,
            nearest_node_id=str(nearest_node_id),
            nearest_node_members=member_names,
            nearest_node_diseases=member_diseases,
            majority_disease=majority_disease,
        )


if __name__ == "__main__":
    space = MapperSpace.build()
    space.save()
    print(f"Mapper space built: {len(space.graph['nodes'])} nodes, "
          f"{sum(len(v) for v in space.graph['links'].values())} edges")
    print("\nTo project a new compound, standardize its raw properties with")
    print("your existing z-score pipeline, then call space.project_query(...).")
