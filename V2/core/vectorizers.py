"""
Gestionnaire de vectorisation modulaire pour Dimensionality Lab
===============================================================

Gère différentes méthodes de vectorisation pour texte et images :
- Texte : TF-IDF, SVD, HuggingFace embeddings, etc.
- Images : StandardScaler, PCA, raw, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import warnings

# Optional imports for advanced vectorizers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class VectorizerManager:
    """Gestionnaire centralisé pour toutes les méthodes de vectorisation."""

    # Méthodes disponibles par type de données
    TEXT_METHODS = [
        "TF-IDF + SVD",           # Par défaut
        "TF-IDF brut",
        "PCA sur TF-IDF",
        "None (brut)",
        "HuggingFace (MiniLM)",   # Sentence transformers
        "HuggingFace (MPNet)",    # Alternative model
    ]

    IMAGE_METHODS = [
        "StandardScaler",         # Par défaut
        "PCA (50)",
        "PCA (100)",
        "None (brut)",
        "MinMaxScaler"
    ]

    def __init__(self):
        """Initialise le gestionnaire de vectorisation."""
        self.vectorizer = None
        self.scaler = None
        self.reducer = None
        self.sentence_model = None
        self.last_method = None
        self.metadata = {}

    def get_methods_for_type(self, dataset_type: str) -> list:
        """
        Retourne les méthodes de vectorisation disponibles selon le type de données.

        Parameters
        ----------
        dataset_type : str
            Type de dataset ('text' ou 'image')

        Returns
        -------
        methods : list
            Liste des méthodes disponibles
        """
        if dataset_type == 'text':
            methods = self.TEXT_METHODS.copy()
            # Filtrer HuggingFace si non disponible
            if not HAS_SENTENCE_TRANSFORMERS:
                methods = [m for m in methods if not m.startswith("HuggingFace")]
            return methods
        else:
            return self.IMAGE_METHODS.copy()

    def vectorize(self, X: np.ndarray, dataset_type: str, method: str,
                  svd_components: int = 300) -> np.ndarray:
        """
        Vectorise les données selon la méthode choisie.

        Parameters
        ----------
        X : np.ndarray
            Données d'entrée (texte ou images)
        dataset_type : str
            Type de dataset ('text' ou 'image')
        method : str
            Méthode de vectorisation à utiliser
        svd_components : int, optional
            Nombre de composantes pour SVD/PCA (par défaut 300)

        Returns
        -------
        X_vectorized : np.ndarray
            Données vectorisées
        """
        self.last_method = method
        self.metadata = {
            'method': method,
            'input_shape': X.shape,
            'dataset_type': dataset_type
        }

        if dataset_type == 'text':
            X_vectorized = self._vectorize_text(X, method, svd_components)
        else:
            X_vectorized = self._vectorize_images(X, method)

        self.metadata['output_shape'] = X_vectorized.shape
        self.metadata['sparsity'] = self._calculate_sparsity(X_vectorized)

        return X_vectorized

    def _vectorize_text(self, X: np.ndarray, method: str, svd_components: int) -> np.ndarray:
        """Vectorise des données textuelles."""

        if method == "TF-IDF + SVD":
            # TF-IDF puis réduction SVD
            if not hasattr(self, 'tfidf_vectorizer'):
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    max_df=0.8,
                    ngram_range=(1, 2)
                )

            # Si X est déjà TF-IDF sparse, on applique directement SVD
            if sparse.issparse(X):
                X_tfidf = X
            else:
                # Assume X contient du texte brut
                if isinstance(X[0], str):
                    X_tfidf = self.tfidf_vectorizer.fit_transform(X)
                else:
                    # X est déjà vectorisé
                    X_tfidf = X

            # Réduction dimensionnelle avec SVD
            n_components = min(svd_components, min(X_tfidf.shape) - 1)
            self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = self.reducer.fit_transform(X_tfidf)

            # Normalisation après SVD
            self.scaler = StandardScaler()
            X_final = self.scaler.fit_transform(X_reduced)

            self.metadata['tfidf_features'] = X_tfidf.shape[1] if hasattr(X_tfidf, 'shape') else None
            self.metadata['svd_components'] = n_components
            self.metadata['explained_variance_ratio'] = self.reducer.explained_variance_ratio_.sum()

            return X_final

        elif method == "TF-IDF brut":
            # TF-IDF sans réduction
            if not hasattr(self, 'tfidf_vectorizer'):
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    max_df=0.8
                )

            if sparse.issparse(X):
                return X.toarray()
            elif isinstance(X[0], str):
                X_tfidf = self.tfidf_vectorizer.fit_transform(X)
                return X_tfidf.toarray()
            else:
                return X

        elif method == "PCA sur TF-IDF":
            # TF-IDF puis PCA (dense)
            if not hasattr(self, 'tfidf_vectorizer'):
                self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

            if sparse.issparse(X):
                X_tfidf = X.toarray()
            elif isinstance(X[0], str):
                X_tfidf = self.tfidf_vectorizer.fit_transform(X).toarray()
            else:
                X_tfidf = X

            n_components = min(svd_components, min(X_tfidf.shape) - 1)
            self.reducer = PCA(n_components=n_components, random_state=42)
            X_pca = self.reducer.fit_transform(X_tfidf)

            self.metadata['pca_components'] = n_components
            self.metadata['explained_variance_ratio'] = self.reducer.explained_variance_ratio_.sum()

            return X_pca

        elif method == "None (brut)":
            # Données brutes sans transformation
            if sparse.issparse(X):
                return X.toarray()
            return X

        elif method.startswith("HuggingFace"):
            if not HAS_SENTENCE_TRANSFORMERS:
                warnings.warn("sentence-transformers non installé. Utilisation de TF-IDF + SVD à la place.")
                return self._vectorize_text(X, "TF-IDF + SVD", svd_components)

            # Sélection du modèle
            if "MiniLM" in method:
                model_name = "all-MiniLM-L6-v2"  # Rapide et léger (384 dims)
            else:  # MPNet
                model_name = "all-mpnet-base-v2"  # Plus précis (768 dims)

            if self.sentence_model is None or self.metadata.get('hf_model') != model_name:
                self.sentence_model = SentenceTransformer(model_name)
                self.metadata['hf_model'] = model_name

            # Conversion en embeddings
            if isinstance(X[0], str):
                embeddings = self.sentence_model.encode(X, show_progress_bar=True)
            else:
                warnings.warn("HuggingFace attend du texte brut. Utilisation de TF-IDF + SVD.")
                return self._vectorize_text(X, "TF-IDF + SVD", svd_components)

            self.metadata['embedding_dim'] = embeddings.shape[1]
            return embeddings

        else:
            warnings.warn(f"Méthode inconnue: {method}. Utilisation de TF-IDF + SVD.")
            return self._vectorize_text(X, "TF-IDF + SVD", svd_components)

    def _vectorize_images(self, X: np.ndarray, method: str) -> np.ndarray:
        """Vectorise des données images."""

        if method == "StandardScaler":
            # Normalisation standard
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled

        elif method.startswith("PCA"):
            # Extraction du nombre de composantes
            if "50" in method:
                n_components = 50
            elif "100" in method:
                n_components = 100
            else:
                n_components = 50  # Par défaut

            n_components = min(n_components, min(X.shape) - 1)

            # Normalisation puis PCA
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.reducer = PCA(n_components=n_components, random_state=42)
            X_pca = self.reducer.fit_transform(X_scaled)

            self.metadata['pca_components'] = n_components
            self.metadata['explained_variance_ratio'] = self.reducer.explained_variance_ratio_.sum()

            return X_pca

        elif method == "None (brut)":
            # Données brutes
            return X

        elif method == "MinMaxScaler":
            # Normalisation min-max [0, 1]
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled

        else:
            warnings.warn(f"Méthode inconnue: {method}. Utilisation de StandardScaler.")
            return self._vectorize_images(X, "StandardScaler")

    def _calculate_sparsity(self, X: np.ndarray) -> float:
        """Calcule le pourcentage de valeurs nulles dans la matrice."""
        if sparse.issparse(X):
            return 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        else:
            return np.sum(X == 0) / X.size

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées de la dernière vectorisation.

        Returns
        -------
        metadata : Dict[str, Any]
            Informations sur la vectorisation (méthode, dimensions, variance, etc.)
        """
        return self.metadata.copy()

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Applique la même transformation à de nouvelles données.

        Parameters
        ----------
        X_new : np.ndarray
            Nouvelles données à transformer

        Returns
        -------
        X_transformed : np.ndarray
            Données transformées avec la même méthode
        """
        if self.last_method is None:
            raise ValueError("Aucune vectorisation n'a été effectuée. Appelez d'abord vectorize().")

        # Appliquer les transformations sauvegardées
        X_transformed = X_new

        if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
            if isinstance(X_new[0], str):
                X_transformed = self.tfidf_vectorizer.transform(X_new)
                if not sparse.issparse(X_transformed):
                    X_transformed = X_transformed

        if self.reducer is not None:
            if sparse.issparse(X_transformed):
                X_transformed = self.reducer.transform(X_transformed)
            else:
                X_transformed = self.reducer.transform(X_transformed)

        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)

        if self.sentence_model is not None and isinstance(X_new[0], str):
            X_transformed = self.sentence_model.encode(X_new)

        return X_transformed

    def reset(self):
        """Réinitialise le gestionnaire de vectorisation."""
        self.vectorizer = None
        self.scaler = None
        self.reducer = None
        self.sentence_model = None
        self.last_method = None
        self.metadata = {}
        if hasattr(self, 'tfidf_vectorizer'):
            delattr(self, 'tfidf_vectorizer')