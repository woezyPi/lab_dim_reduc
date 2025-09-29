"""
Gestionnaire de clustering pour UMAP vs t-SNE Explorer
======================================================

Implémente les méthodes de clustering sur les embeddings.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from typing import Dict, Any, Optional


class ClusteringManager:
    """Gestionnaire pour le clustering des embeddings."""

    def __init__(self):
        """Initialise le gestionnaire de clustering."""
        pass

    def cluster_embeddings(self, X_embedded: np.ndarray, y_true: np.ndarray,
                          n_clusters: int) -> Dict[str, Any]:
        """
        Applique le clustering sur les embeddings et évalue la qualité.

        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, 2)
            Embeddings 2D
        y_true : ndarray of shape (n_samples,)
            Labels vrais
        n_clusters : int
            Nombre de clusters attendu

        Returns
        -------
        results : Dict[str, Any]
            Résultats du clustering avec métriques
        """
        best_score = -1
        best_algorithm = None
        best_labels = None

        # KMeans
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_embedded)
            kmeans_score = silhouette_score(X_embedded, kmeans_labels)

            if kmeans_score > best_score:
                best_score = kmeans_score
                best_algorithm = 'kmeans'
                best_labels = kmeans_labels
        except:
            pass

        # Agglomerative
        try:
            agglo = AgglomerativeClustering(n_clusters=n_clusters)
            agglo_labels = agglo.fit_predict(X_embedded)
            agglo_score = silhouette_score(X_embedded, agglo_labels)

            if agglo_score > best_score:
                best_score = agglo_score
                best_algorithm = 'agglomerative'
                best_labels = agglo_labels
        except:
            pass

        # DBSCAN (auto eps)
        try:
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(X_embedded)
            distances, indices = neighbors_fit.kneighbors(X_embedded)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 90)

            dbscan = DBSCAN(eps=eps, min_samples=4)
            dbscan_labels = dbscan.fit_predict(X_embedded)

            if len(np.unique(dbscan_labels[dbscan_labels != -1])) > 1:
                dbscan_score = silhouette_score(X_embedded[dbscan_labels != -1],
                                               dbscan_labels[dbscan_labels != -1])
                if dbscan_score > best_score:
                    best_score = dbscan_score
                    best_algorithm = 'dbscan'
                    best_labels = dbscan_labels
        except:
            pass

        # Calcul des métriques si clustering réussi
        if best_labels is not None:
            ari = adjusted_rand_score(y_true, best_labels)
            nmi = normalized_mutual_info_score(y_true, best_labels)

            return {
                'best_algorithm': best_algorithm,
                'n_clusters': len(np.unique(best_labels[best_labels != -1])),
                'silhouette_score': best_score,
                'ari': ari,
                'nmi': nmi,
                'labels': best_labels
            }

        return {
            'best_algorithm': 'none',
            'n_clusters': 0,
            'silhouette_score': 0,
            'ari': 0,
            'nmi': 0,
            'labels': None
        }