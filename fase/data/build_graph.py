"""
fase/data/build_graph.py
========================
Phase 2 — Graph Construction

Builds two complementary adjacency matrices for the 25 Baltimore ZCTA nodes
and merges them into a single weighted edge list in PyTorch Geometric format.

A_geo  (geographic contiguity)
    Constructed with libpysal Queen contiguity: two ZCTAs are adjacent when
    their polygons share any boundary (edge or vertex).  Captures direct
    spatial spillover.  Weights are row-normalised (stochastic, D^-1 A).
    Isolated nodes (if any) fall back to K-nearest centroids.

A_feat  (demographic feature similarity)
    Cosine similarity of each ZCTA's ACS feature vector
    [pct_minority, median_income_norm, poverty_rate].
    Captures structural similarity between non-adjacent but demographically
    comparable areas.  Entries below `feat_sim_threshold` are zeroed out.

Combined adjacency
    A = alpha_geo * A_geo_norm + alpha_feat * A_feat_thresh

    Self-loops are added to the diagonal (= 1.0) after combination so that
    every node aggregates its own features in the GNN message-passing step.

Output artefacts written to data/tensors/baltimore/
----------------------------------------------------
graph.pkl  — Python pickle dict with keys:
    edge_index      LongTensor  (2, E)    source / target node indices
    edge_weight     FloatTensor (E,)      combined edge weights
    num_nodes       int                   25
    node_ids        list[str]             ZCTA5 codes in tensor row order
    A_geo           ndarray (N, N)        raw binary Queen adjacency
    A_geo_norm      ndarray (N, N)        row-normalised geo adjacency
    A_feat          ndarray (N, N)        raw cosine similarity matrix
    A_feat_thresh   ndarray (N, N)        thresholded feat similarity
    A_combined      ndarray (N, N)        final combined adjacency

Usage
-----
    python -m fase.data.build_graph
    python -m fase.data.build_graph --geo-type knn --knn-k 5
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from libpysal.weights import Queen, KNN

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
DATA_TENSORS = ROOT / "data" / "tensors"
META_PKL     = DATA_TENSORS / "baltimore" / "meta.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load inputs from Phase 1 metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_inputs(
    meta_path: Path = META_PKL,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, List[str]]:
    """
    Load the ZCTA GeoDataFrame, ACS feature table, and canonical node order
    from the Phase 1 meta.pkl artefact.

    Returns
    -------
    zcta_gdf  : GeoDataFrame (N rows) with columns ZCTA5, geometry.
                Filtered and re-ordered to match `node_ids`.
    acs_df    : DataFrame (N rows) with columns ZCTA5, pct_minority,
                median_income_norm, poverty_rate. Same ordering.
    node_ids  : list[str] of ZCTA5 codes (length N) — canonical node order.
    """
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Phase 1 metadata not found at {meta_path}.\n"
            "Run Phase 1 first:  python -m fase.data.build_tensors"
        )
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)

    node_ids: List[str] = meta["node_ids"]
    N = len(node_ids)

    # Filter and reorder zcta_gdf to the canonical node list.
    # Phase 1 may retain more ZCTAs in the GDF than ended up in the tensor
    # (e.g. ZCTAs with zero incidents are excluded from node_ids).
    zcta_full: gpd.GeoDataFrame = meta["zcta_gdf"]
    zcta_gdf = (
        zcta_full.set_index("ZCTA5")
        .loc[node_ids]          # reorder to canonical node order
        .reset_index()
    )

    # ACS — reindex to canonical order, zero-fill any missing ZCTAs
    acs_full: pd.DataFrame = meta["acs"]
    acs_df = (
        acs_full.set_index("ZCTA5")
        .reindex(node_ids)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "ZCTA5"})
    )

    logger.info(
        "Loaded %d ZCTA nodes from %s.", N, meta_path.name
    )
    return zcta_gdf, acs_df, node_ids


# ─────────────────────────────────────────────────────────────────────────────
# 2. Geographic adjacency  (A_geo)
# ─────────────────────────────────────────────────────────────────────────────

def build_geo_adjacency(
    zcta_gdf: gpd.GeoDataFrame,
    node_ids: List[str],
    geo_type: str = "queen",
    knn_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the binary geographic adjacency matrix and its row-normalised form.

    Strategy
    --------
    * Default: Queen contiguity (libpysal).  Two ZCTAs are adjacent if their
      polygons share any boundary (edge or point).
    * Fallback for isolated nodes (degree 0 after Queen): supplement with the
      k nearest centroids so the graph stays connected.
    * Alternative: pass geo_type='knn' to use only centroid-KNN.

    Parameters
    ----------
    zcta_gdf : GeoDataFrame (N rows) in EPSG:4326
    node_ids : canonical node order (must match zcta_gdf rows)
    geo_type : 'queen' or 'knn'
    knn_k    : number of neighbours for KNN (used as fallback or primary)

    Returns
    -------
    A_geo      : (N, N) float32 binary adjacency (symmetric, no self-loops)
    A_geo_norm : (N, N) float32 row-normalised adjacency (D^-1 A)
    """
    N = len(node_ids)

    # Work in a projected CRS (metres) for accurate centroid / KNN distance
    zcta_proj = zcta_gdf.to_crs("EPSG:32618")  # UTM Zone 18N (Maryland)

    if geo_type == "knn":
        w = KNN.from_dataframe(zcta_proj, k=knn_k, use_index=False)
        logger.info("Built KNN(%d) adjacency.", knn_k)
    else:
        w = Queen.from_dataframe(zcta_proj, use_index=False, silence_warnings=True)
        n_islands = len(w.islands)
        logger.info(
            "Built Queen contiguity adjacency.  Islands (degree-0 nodes): %d",
            n_islands,
        )
        if n_islands > 0:
            # Supplement islands with their nearest k centroid neighbours
            logger.info(
                "Supplementing %d isolated ZCTA(s) with KNN(%d) edges.",
                n_islands, knn_k,
            )
            w_knn = KNN.from_dataframe(zcta_proj, k=knn_k, use_index=False)
            for island_idx in w.islands:
                for nb_idx, nb_w in zip(
                    w_knn.neighbors[island_idx],
                    w_knn.weights[island_idx],
                ):
                    w.neighbors[island_idx].append(nb_idx)
                    w.weights[island_idx].append(nb_w)
                    # Ensure symmetry
                    if island_idx not in w.neighbors[nb_idx]:
                        w.neighbors[nb_idx].append(island_idx)
                        w.weights[nb_idx].append(nb_w)

    # Build dense binary (N, N) matrix ordered by node_ids
    # libpysal indexes by the integer position of each row in the GDF
    A_geo = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in w.neighbors[i]:
            A_geo[i, j] = 1.0

    # Symmetrise (Queen should already be symmetric, but be safe)
    A_geo = np.maximum(A_geo, A_geo.T)

    # Row-normalise: D^-1 A
    row_sums = A_geo.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)   # guard isolated nodes
    A_geo_norm = (A_geo / row_sums).astype(np.float32)

    n_edges = int(A_geo.sum())
    logger.info(
        "A_geo: %d directed edges  (avg degree %.2f).",
        n_edges, n_edges / N,
    )
    return A_geo, A_geo_norm


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature-similarity adjacency  (A_feat)
# ─────────────────────────────────────────────────────────────────────────────

def build_feat_adjacency(
    acs_df: pd.DataFrame,
    node_ids: List[str],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a demographic feature-similarity adjacency matrix via cosine
    similarity on the ACS feature vectors, then threshold to sparsify.

    Features used: pct_minority, median_income_norm, poverty_rate  (dim = 3)

    Parameters
    ----------
    acs_df    : DataFrame with ZCTA5 + feature columns, ordered by node_ids
    node_ids  : canonical node order
    threshold : cosine-similarity cutoff; pairs below this value get weight 0

    Returns
    -------
    A_feat        : (N, N) float32 raw cosine similarity  (diagonal = 1)
    A_feat_thresh : (N, N) float32 thresholded matrix     (diagonal = 0)
    """
    feat_cols = ["pct_minority", "median_income_norm", "poverty_rate"]
    F = (
        acs_df.set_index("ZCTA5")
        .reindex(node_ids)[feat_cols]
        .fillna(0.0)
        .values
        .astype(np.float32)
    )   # (N, 3)

    # L2-normalise rows; protect zero vectors with epsilon
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    F_norm = F / norms

    # Cosine similarity matrix
    A_feat = (F_norm @ F_norm.T).astype(np.float32)   # (N, N)

    # Threshold and remove self-similarity
    A_feat_thresh = np.where(A_feat >= threshold, A_feat, 0.0).astype(np.float32)
    np.fill_diagonal(A_feat_thresh, 0.0)

    n_edges = int((A_feat_thresh > 0).sum())
    logger.info(
        "A_feat: threshold=%.2f  %d directed edges  (avg degree %.2f).",
        threshold, n_edges, n_edges / len(node_ids),
    )
    return A_feat, A_feat_thresh


# ─────────────────────────────────────────────────────────────────────────────
# 4. Combine adjacencies
# ─────────────────────────────────────────────────────────────────────────────

def combine_adjacencies(
    A_geo_norm: np.ndarray,
    A_feat_thresh: np.ndarray,
    alpha_geo: float = 0.6,
    alpha_feat: float = 0.4,
    self_loops: bool = True,
) -> np.ndarray:
    """
    Merge geographic and feature-similarity adjacencies into a single matrix.

        A_combined = alpha_geo * A_geo_norm + alpha_feat * A_feat_thresh

    Self-loops (diagonal = 1.0) are added last when self_loops=True.
    This lets each node aggregate its own representation during GNN
    message-passing without diluting the neighbour weights.

    Returns
    -------
    A_combined : (N, N) float32
    """
    A = alpha_geo * A_geo_norm + alpha_feat * A_feat_thresh
    if self_loops:
        np.fill_diagonal(A, 1.0)
    n_edges = int((A > 0).sum())
    logger.info(
        "A_combined: alpha_geo=%.2f  alpha_feat=%.2f  self_loops=%s  "
        "%d non-zero entries.",
        alpha_geo, alpha_feat, self_loops, n_edges,
    )
    return A.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convert to PyG edge_index / edge_weight
# ─────────────────────────────────────────────────────────────────────────────

def to_edge_index(
    A: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense adjacency matrix to the sparse COO format expected by
    PyTorch Geometric.

    Parameters
    ----------
    A : (N, N) float32 adjacency matrix

    Returns
    -------
    edge_index  : LongTensor  (2, E)   — row 0 = sources, row 1 = targets
    edge_weight : FloatTensor (E,)
    """
    rows, cols = np.nonzero(A)
    edge_index  = torch.tensor(
        np.stack([rows, cols], axis=0), dtype=torch.long
    )
    edge_weight = torch.tensor(A[rows, cols], dtype=torch.float32)
    logger.info(
        "PyG edge_index: shape=%s  edge_weight: shape=%s  "
        "weight range [%.4f, %.4f].",
        tuple(edge_index.shape), tuple(edge_weight.shape),
        float(edge_weight.min()), float(edge_weight.max()),
    )
    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_ids: List[str],
    A_geo: np.ndarray,
    A_geo_norm: np.ndarray,
    A_feat: np.ndarray,
    A_feat_thresh: np.ndarray,
    A_combined: np.ndarray,
    out_dir: Path = DATA_TENSORS,
) -> Path:
    """
    Persist the graph to ``out_dir/baltimore/graph.pkl``.

    Saved keys
    ----------
    edge_index      LongTensor  (2, E)
    edge_weight     FloatTensor (E,)
    num_nodes       int
    node_ids        list[str]
    A_geo           ndarray (N, N)  raw Queen binary adjacency
    A_geo_norm      ndarray (N, N)  row-normalised geo adjacency
    A_feat          ndarray (N, N)  raw cosine similarity
    A_feat_thresh   ndarray (N, N)  thresholded cosine similarity
    A_combined      ndarray (N, N)  final combined adjacency
    """
    out = out_dir / "baltimore"
    out.mkdir(parents=True, exist_ok=True)

    graph = {
        "edge_index":    edge_index,
        "edge_weight":   edge_weight,
        "num_nodes":     len(node_ids),
        "node_ids":      node_ids,
        "A_geo":         A_geo,
        "A_geo_norm":    A_geo_norm,
        "A_feat":        A_feat,
        "A_feat_thresh": A_feat_thresh,
        "A_combined":    A_combined,
    }
    out_path = out / "graph.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved graph.pkl  →  %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def log_graph_stats(
    A_geo: np.ndarray,
    A_feat_thresh: np.ndarray,
    A_combined: np.ndarray,
    node_ids: List[str],
) -> None:
    """Print a per-node degree summary for quick sanity checking."""
    N = len(node_ids)
    geo_deg  = A_geo.sum(axis=1)
    feat_deg = (A_feat_thresh > 0).sum(axis=1)
    comb_deg = (A_combined > 0).sum(axis=1) - 1   # subtract self-loop

    logger.info(
        "%-8s  %6s  %9s  %9s",
        "ZCTA5", "geo_deg", "feat_deg", "comb_deg",
    )
    for i, z in enumerate(node_ids):
        logger.info(
            "%-8s  %6.0f  %9.0f  %9.0f",
            z, geo_deg[i], feat_deg[i], comb_deg[i],
        )

    logger.info(
        "Summary — geo: min=%d max=%d mean=%.1f | "
        "feat: min=%d max=%d mean=%.1f | "
        "combined: min=%d max=%d mean=%.1f",
        geo_deg.min(), geo_deg.max(), geo_deg.mean(),
        feat_deg.min(), feat_deg.max(), feat_deg.mean(),
        comb_deg.min(), comb_deg.max(), comb_deg.mean(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build(
    meta_path: Path = META_PKL,
    geo_type: str = "queen",
    knn_k: int = 5,
    feat_threshold: float = 0.5,
    alpha_geo: float = 0.6,
    alpha_feat: float = 0.4,
    self_loops: bool = True,
    out_dir: Path = DATA_TENSORS,
) -> Dict:
    """
    Full Phase-2 pipeline.  Returns the graph dict for downstream use.

    Parameters
    ----------
    meta_path      : path to Phase 1 meta.pkl
    geo_type       : 'queen' (default) or 'knn' for geographic adjacency
    knn_k          : K for KNN fallback / primary (used only when relevant)
    feat_threshold : cosine-similarity cutoff for A_feat
    alpha_geo      : weight of geo adjacency in combined matrix
    alpha_feat     : weight of feat adjacency in combined matrix
    self_loops     : whether to add self-loops (diagonal = 1) to A_combined
    out_dir        : root output dir; graph saved under out_dir/baltimore/
    """
    logger.info("=" * 60)
    logger.info("FASE Phase 2 — graph construction")
    logger.info(
        "  geo_type=%s  feat_threshold=%.2f  "
        "alpha=(geo=%.2f, feat=%.2f)  self_loops=%s",
        geo_type, feat_threshold, alpha_geo, alpha_feat, self_loops,
    )
    logger.info("=" * 60)

    zcta_gdf, acs_df, node_ids = load_graph_inputs(meta_path)

    A_geo, A_geo_norm       = build_geo_adjacency(
        zcta_gdf, node_ids, geo_type=geo_type, knn_k=knn_k
    )
    A_feat, A_feat_thresh   = build_feat_adjacency(
        acs_df, node_ids, threshold=feat_threshold
    )
    A_combined              = combine_adjacencies(
        A_geo_norm, A_feat_thresh,
        alpha_geo=alpha_geo, alpha_feat=alpha_feat, self_loops=self_loops,
    )
    edge_index, edge_weight = to_edge_index(A_combined)

    log_graph_stats(A_geo, A_feat_thresh, A_combined, node_ids)

    save_graph(
        edge_index, edge_weight, node_ids,
        A_geo, A_geo_norm, A_feat, A_feat_thresh, A_combined,
        out_dir=out_dir,
    )

    logger.info("Phase 2 complete.")
    return {
        "edge_index": edge_index, "edge_weight": edge_weight,
        "num_nodes":  len(node_ids), "node_ids": node_ids,
        "A_geo": A_geo, "A_geo_norm": A_geo_norm,
        "A_feat": A_feat, "A_feat_thresh": A_feat_thresh,
        "A_combined": A_combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FASE Phase 2: build spatial graph from Baltimore ZCTAs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--meta-path", type=Path, default=META_PKL,
        help="Path to Phase 1 meta.pkl.",
    )
    parser.add_argument(
        "--geo-type", choices=["queen", "knn"], default="queen",
        help="Geographic adjacency type.",
    )
    parser.add_argument(
        "--knn-k", type=int, default=5,
        help="K for KNN adjacency (used as fallback or primary).",
    )
    parser.add_argument(
        "--feat-threshold", type=float, default=0.5,
        help="Cosine-similarity threshold for feature graph.",
    )
    parser.add_argument(
        "--alpha-geo", type=float, default=0.6,
        help="Weight of geographic adjacency in combined matrix.",
    )
    parser.add_argument(
        "--alpha-feat", type=float, default=0.4,
        help="Weight of feature adjacency in combined matrix.",
    )
    parser.add_argument(
        "--no-self-loops", action="store_true",
        help="Omit self-loops from the combined adjacency.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DATA_TENSORS,
        help="Root output directory.",
    )
    args = parser.parse_args()

    build(
        meta_path=args.meta_path,
        geo_type=args.geo_type,
        knn_k=args.knn_k,
        feat_threshold=args.feat_threshold,
        alpha_geo=args.alpha_geo,
        alpha_feat=args.alpha_feat,
        self_loops=not args.no_self_loops,
        out_dir=args.out_dir,
    )
