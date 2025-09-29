# core/embeddings.py
# ===================
# Gestion des pré-traitements et calculs d'embeddings (UMAP / t-SNE)
# Compatible avec l'app v4 : prépare X (texte/images), puis calcule UMAP/t-SNE
# sans densifier le TF-IDF. Supporte les paramètres avancés UMAP et la supervision.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap  # type: ignore


@dataclass
class EmbeddingManager:
    """
    Gestionnaire de pipelines d'embedding.

    Notes
    -----
    - Pour le texte (20 Newsgroups, TF-IDF), on applique:
        TF-IDF (sparse) -> TruncatedSVD(n_components=k) -> StandardScaler -> (UMAP/t-SNE)
      On ne densifie jamais TF-IDF : pas de .toarray() ici.
    - Pour les images (Digits), on applique StandardScaler puis on passe à UMAP/t-SNE.
    - compute_umap supporte les paramètres avancés (a/b, densmap, etc.) et la supervision
      via target_metric='categorical' si `y` est fourni.
    - compute_tsne propose une PCA(30) de pré-normalisation optionnelle (pca_prenorm=True).
    """

    # caches optionnels (réutilisation dans une même session, si souhaité plus tard)
    svd_: Optional[TruncatedSVD] = None
    scaler_: Optional[StandardScaler] = None
    pca_prenorm_: Optional[PCA] = None

    # ========================
    # Préparation des données
    # ========================
    def prepare_text_data(self, X: np.ndarray, svd_components: int = 300) -> np.ndarray:
        """
        Prépare des données texte vectorisées TF-IDF (sparse) pour UMAP/t-SNE.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrice TF-IDF (souvent CSR sparse). Peut être dense si déjà compressée.
        svd_components : int, default=300
            Nombre de composantes pour TruncatedSVD (LSA).

        Returns
        -------
        X_proc : ndarray, shape (n_samples, svd_components)
            Représentation dense mais compacte, standardisée.
        """
        if sparse.issparse(X):
            svd = TruncatedSVD(n_components=svd_components, random_state=42)
            X_red = svd.fit_transform(X)  # dense (n_samples, k)
            self.svd_ = svd
        else:
            # Si déjà dense (ex: déjà SVD), ne refais pas un SVD si dim <= svd_components
            if X.shape[1] > svd_components:
                svd = TruncatedSVD(n_components=svd_components, random_state=42)
                X_red = svd.fit_transform(X)
                self.svd_ = svd
            else:
                X_red = X  # déjà compact
                self.svd_ = None

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X_red)
        self.scaler_ = scaler
        return X_proc

    def prepare_image_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prépare des données images (ex: Digits) pour UMAP/t-SNE.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Données denses (pixels/traits).

        Returns
        -------
        X_proc : ndarray
            Données standardisées (z-score).
        """
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X)
        self.scaler_ = scaler
        return X_proc

    # ========================
    # UMAP
    # ========================
    def compute_umap(
        self, X: np.ndarray, params: Dict[str, Any], y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, umap.UMAP]:
        """
        Calcule un embedding UMAP avec paramètres avancés + supervision optionnelle.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Données d'entrée (déjà préparées : texte->SVD, images->scaler).
        params : dict
            Dictionnaire d'hyperparamètres UMAP. Exemples clés acceptés :
            - n_neighbors, min_dist, spread, n_epochs
            - metric, init, negative_sample_rate, transform_queue_size
            - angular_rp_forest, set_op_mix_ratio, local_connectivity
            - repulsion_strength, densmap, output_metric, low_memory
            - random_state, a, b
            - target_metric: 'categorical' pour supervision (si y fourni)
        y : ndarray, optional
            Labels pour supervision (utilisé si params['target_metric'] == 'categorical').

        Returns
        -------
        X_umap : ndarray, shape (n_samples, n_components)
            Embedding UMAP.
        model : umap.UMAP
            Modèle UMAP entraîné.
        """
        # Séparer et nettoyer les paramètres
        params = dict(params or {})

        # Gestion a/b: beaucoup de configs utilisent 0 pour "auto"
        for key in ("a", "b"):
            if key in params and (params[key] is None or (isinstance(params[key], (int, float)) and params[key] == 0)):
                params[key] = None

        # Supervision
        target_metric = params.pop("target_metric", None)
        fit_y = y if target_metric == "categorical" and y is not None else None

        # Filtrer None (umap n'aime pas recevoir des kwargs=None pour certains params)
        clean_params = {k: v for k, v in params.items() if v is not None}

        model = umap.UMAP(**clean_params)
        X_umap = model.fit_transform(X, y=fit_y)
        return X_umap, model

    # ========================
    # t-SNE
    # ========================
    def compute_tsne(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, TSNE]:
        """
        Calcule un embedding t-SNE avec PCA pré-normalisation optionnelle.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Données d'entrée (déjà préparées).
        params : dict
            - perplexity : int
            - learning_rate : {'auto'} ou float
            - max_iter : int (itérations)
            - pca_prenorm : bool (si True, applique PCA(30) avant t-SNE)

        Returns
        -------
        X_tsne : ndarray, shape (n_samples, 2)
            Embedding t-SNE (2D).
        model : TSNE
            Modèle t-SNE entraîné.
        """
        perplexity = int(params.get("perplexity", 30))
        learning_rate = params.get("learning_rate", "auto")
        max_iter = int(params.get("max_iter", 1000))
        pca_prenorm = bool(params.get("pca_prenorm", True))

        n_samples = X.shape[0]
        # Règle t-SNE: perplexity < (n_samples - 1) / 3
        max_perp = (n_samples - 1) / 3.0
        if perplexity >= max_perp:
            raise ValueError(
                f"t-SNE 'perplexity' trop élevée pour n={n_samples}. "
                f"Requis: perplexity < {max_perp:.1f}, reçu: {perplexity}."
            )

        X_in = X
        # PCA(30) avant t-SNE pour stabilité et vitesse
        if pca_prenorm and X.shape[1] > 30:
            self.pca_prenorm_ = PCA(n_components=30, random_state=42)
            X_in = self.pca_prenorm_.fit_transform(X)
        else:
            self.pca_prenorm_ = None

        # Compatibilité arrière : certaines vieilles versions utilisaient n_iter
        def _make_tsne(**kw):
            try:
                return TSNE(**kw)  # scikit-learn >= 1.5 (max_iter supporté)
            except TypeError:
                kw.pop("max_iter", None)
                kw["n_iter"] = max_iter  # fallback anciennes versions
                return TSNE(**kw)

        model = _make_tsne(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,  # clé moderne
            random_state=42,
        )
        X_tsne = model.fit_transform(X_in)
        return X_tsne, model
