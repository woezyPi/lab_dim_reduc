# core/utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import streamlit as st
import random
import os

def seed_everything(seed: int = 42):
    import torch as _torch  # if installed; ignore if not
    random.seed(seed)
    np.random.seed(seed)
    try:
        _torch.manual_seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as _tf  # noqa
        _tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass
class UIUtils:
    """Small helpers for the Streamlit UI (safe: no help/write of containers)."""

    def validate_parameters(self, params: Dict[str, Dict[str, Any]], n_samples: int) -> List[str]:
        warns = []
        # t-SNE perplexity rule
        perp = int(params["tsne"]["perplexity"])
        if perp >= (n_samples - 1) / 3:
            warns.append(f"t-SNE: perplexity ({perp}) doit être < {(n_samples - 1)//3}.")
        # UMAP min_dist sanity
        md = float(params["umap"]["min_dist"])
        if md < 0 or md > 1:
            warns.append("UMAP: min_dist doit être dans [0,1].")
        return warns

    def display_metrics_card(self, title: str, metrics: Dict[str, float], clustering: Optional[Dict[str, float]] = None) -> None:
        st.markdown(f"**{title}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Trustworthiness", f"{metrics.get('trustworthiness', float('nan')):.3f}")
        with c2:
            st.metric("KNN Accuracy", f"{metrics.get('knn_accuracy', float('nan')):.3f}")
        with c3:
            st.metric("Shepard r", f"{metrics.get('shepard_correlation', float('nan')):.3f}")
        if clustering:
            st.caption(
                f"Silhouette={clustering.get('silhouette_score', float('nan')):.3f} | "
                f"ARI={clustering.get('ari_score', float('nan')):.3f} | "
                f"NMI={clustering.get('nmi_score', float('nan')):.3f}"
            )
        # Important: do not return containers; return None explicitly.
        return None

    def create_comparison_table(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float],
        cl1: Optional[Dict[str, float]],
        cl2: Optional[Dict[str, float]],
        t1: float,
        t2: float,
    ):
        rows = [
            {"Métrique": "Temps (s)", "UMAP": t1, "t-SNE": t2},
            {"Métrique": "Trustworthiness", "UMAP": metrics1.get("trustworthiness"), "t-SNE": metrics2.get("trustworthiness")},
            {"Métrique": "KNN Accuracy", "UMAP": metrics1.get("knn_accuracy"), "t-SNE": metrics2.get("knn_accuracy")},
            {"Métrique": "Shepard r", "UMAP": metrics1.get("shepard_correlation"), "t-SNE": metrics2.get("shepard_correlation")},
        ]
        if cl1 and cl2:
            rows += [
                {"Métrique": "Silhouette", "UMAP": cl1.get("silhouette_score"), "t-SNE": cl2.get("silhouette_score")},
                {"Métrique": "ARI", "UMAP": cl1.get("ari_score"), "t-SNE": cl2.get("ari_score")},
                {"Métrique": "NMI", "UMAP": cl1.get("nmi_score"), "t-SNE": cl2.get("nmi_score")},
            ]
        import pandas as pd
        df = pd.DataFrame(rows)
        return df
