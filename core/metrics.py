"""
Calculateur de métriques pour UMAP vs t-SNE Explorer - Version Complète
======================================================================

Implémente un ensemble complet de métriques d'évaluation pour embeddings :
- Préservation de structure : trustworthiness, continuity
- Corrélation : Shepard (Pearson entre distances HD/LD)
- Classification : KNN-accuracy avec validation croisée
- Clustering : ARI, NMI, Silhouette après clustering
- Métriques de stress et distorsions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    pairwise_distances, 
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import warnings

from .utils import seed_everything

class MetricsCalculator:
    """Calculateur centralisé pour toutes les métriques d'évaluation."""
    
    def __init__(self):
        """Initialise le calculateur de métriques."""
        seed_everything(42)
    
    def calculate_all_metrics(self, X_high: np.ndarray, 
                             X_low: np.ndarray, 
                             y: np.ndarray,
                             k_neighbors: int = 10) -> Dict[str, float]:
        """
        Calcule toutes les métriques d'évaluation pour un embedding.
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features_high)
            Données dans l'espace de haute dimension
        X_low : ndarray of shape (n_samples, n_features_low)
            Données embedées (basse dimension)
        y : ndarray of shape (n_samples,)
            Labels vrais
        k_neighbors : int, default=10
            Nombre de voisins pour trustworthiness/continuity
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionnaire de toutes les métriques calculées
        """
        metrics = {}
        
        # Métriques de préservation de structure
        try:
            metrics['trustworthiness'] = self.calculate_trustworthiness(
                X_high, X_low, k_neighbors
            )
            metrics['continuity'] = self.calculate_continuity(
                X_high, X_low, k_neighbors
            )
        except Exception as e:
            warnings.warn(f"Erreur métriques structure: {str(e)}")
            metrics['trustworthiness'] = 0.0
            metrics['continuity'] = 0.0
        
        # Corrélation de Shepard
        try:
            metrics['shepard_correlation'] = self.calculate_shepard_correlation(
                X_high, X_low
            )
        except Exception as e:
            warnings.warn(f"Erreur corrélation Shepard: {str(e)}")
            metrics['shepard_correlation'] = 0.0
        
        # Précision KNN
        try:
            metrics['knn_accuracy'] = self.calculate_knn_accuracy(X_low, y)
        except Exception as e:
            warnings.warn(f"Erreur KNN accuracy: {str(e)}")
            metrics['knn_accuracy'] = 0.0
        
        # Métriques de clustering (seront mises à jour par ClusteringManager)
        metrics['silhouette_score'] = 0.0
        metrics['ari_score'] = 0.0
        metrics['nmi_score'] = 0.0
        
        # Métriques de stress
        try:
            metrics['stress'] = self.calculate_stress(X_high, X_low)
            metrics['normalized_stress'] = self.calculate_normalized_stress(X_high, X_low)
        except Exception as e:
            warnings.warn(f"Erreur métriques stress: {str(e)}")
            metrics['stress'] = np.inf
            metrics['normalized_stress'] = np.inf
        
        return metrics
    
    def calculate_trustworthiness(self, X_high: np.ndarray, 
                                 X_low: np.ndarray,
                                 k: int = 10) -> float:
        """
        Calcule la trustworthiness (préservation du voisinage local).
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        k : int, default=10
            Nombre de voisins à considérer
            
        Returns
        -------
        trust : float
            Score de trustworthiness [0, 1], plus élevé = meilleur
        """
        k = min(k, X_high.shape[0] - 1)
        return trustworthiness(X_high, X_low, n_neighbors=k, metric='euclidean')
    
    def calculate_continuity(self, X_high: np.ndarray, 
                           X_low: np.ndarray,
                           k: int = 10) -> float:
        """
        Calcule la continuité (inverse de trustworthiness).
        
        Mesure si les voisins dans l'espace de haute dimension 
        restent voisins dans l'embedding.
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        k : int, default=10
            Nombre de voisins à considérer
            
        Returns
        -------
        continuity : float
            Score de continuité [0, 1], plus élevé = meilleur
        """
        # La continuité est la trustworthiness calculée dans l'autre sens
        k = min(k, X_low.shape[0] - 1)
        return trustworthiness(X_low, X_high, n_neighbors=k, metric='euclidean')
    
    def calculate_shepard_correlation(self, X_high: np.ndarray, 
                                    X_low: np.ndarray,
                                    sample_size: int = 2000) -> float:
        """
        Calcule la corrélation de Shepard entre distances HD et LD.
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        sample_size : int, default=2000
            Nombre d'échantillons pour accélérer le calcul
            
        Returns
        -------
        correlation : float
            Corrélation de Pearson entre distances [-1, 1]
        """
        n_samples = X_high.shape[0]
        
        # Sous-échantillonnage pour performances
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_h_sample = X_high[indices]
            X_l_sample = X_low[indices]
        else:
            X_h_sample = X_high
            X_l_sample = X_low
        
        # Calcul des distances
        dist_high = pdist(X_h_sample, metric='euclidean')
        dist_low = pdist(X_l_sample, metric='euclidean')
        
        # Corrélation de Pearson
        correlation, _ = pearsonr(dist_high, dist_low)
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_knn_accuracy(self, X_embedded: np.ndarray, 
                              y: np.ndarray,
                              k: int = 5,
                              cv_folds: int = 3) -> float:
        """
        Calcule la précision KNN avec validation croisée sur l'embedding.
        
        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, n_components)
            Données embedées
        y : ndarray of shape (n_samples,)
            Labels vrais
        k : int, default=5
            Nombre de voisins pour KNN
        cv_folds : int, default=3
            Nombre de folds pour validation croisée
            
        Returns
        -------
        accuracy : float
            Précision moyenne de la validation croisée [0, 1]
        """
        k = min(k, X_embedded.shape[0] - 1)
        
        # Validation croisée avec KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        try:
            scores = cross_val_score(
                knn, X_embedded, y, 
                cv=min(cv_folds, X_embedded.shape[0] // 2),
                scoring='accuracy'
            )
            return np.mean(scores)
        except Exception as e:
            warnings.warn(f"Erreur KNN CV: {str(e)}")
            return 0.0
    
    def calculate_stress(self, X_high: np.ndarray, 
                        X_low: np.ndarray,
                        sample_size: int = 1000) -> float:
        """
        Calcule le stress (distorsion des distances).
        
        Stress = sum((d_high - d_low)²) / sum(d_high²)
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        sample_size : int, default=1000
            Nombre d'échantillons pour accélérer
            
        Returns
        -------
        stress : float
            Valeur de stress, plus faible = meilleur
        """
        n_samples = X_high.shape[0]
        
        # Sous-échantillonnage
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_h_sample = X_high[indices]
            X_l_sample = X_low[indices]
        else:
            X_h_sample = X_high
            X_l_sample = X_low
        
        # Distances
        dist_high = pdist(X_h_sample, metric='euclidean')
        dist_low = pdist(X_l_sample, metric='euclidean')
        
        # Calcul du stress
        numerator = np.sum((dist_high - dist_low) ** 2)
        denominator = np.sum(dist_high ** 2)
        
        if denominator == 0:
            return np.inf
        
        return numerator / denominator
    
    def calculate_normalized_stress(self, X_high: np.ndarray, 
                                   X_low: np.ndarray,
                                   sample_size: int = 1000) -> float:
        """
        Calcule le stress normalisé.
        
        Normalized Stress = sqrt(sum((d_high - d_low)²) / sum((d_high - mean(d_high))²))
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension  
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        sample_size : int, default=1000
            Nombre d'échantillons pour accélérer
            
        Returns
        -------
        normalized_stress : float
            Stress normalisé, plus faible = meilleur
        """
        n_samples = X_high.shape[0]
        
        # Sous-échantillonnage
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_h_sample = X_high[indices]
            X_l_sample = X_low[indices]
        else:
            X_h_sample = X_high
            X_l_sample = X_low
        
        # Distances
        dist_high = pdist(X_h_sample, metric='euclidean')
        dist_low = pdist(X_l_sample, metric='euclidean')
        
        # Stress normalisé
        mean_dist_high = np.mean(dist_high)
        
        numerator = np.sum((dist_high - dist_low) ** 2)
        denominator = np.sum((dist_high - mean_dist_high) ** 2)
        
        if denominator == 0:
            return np.inf
        
        return np.sqrt(numerator / denominator)
    
    def calculate_cluster_metrics(self, X_embedded: np.ndarray, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques de clustering.
        
        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, n_components)
            Données embedées
        y_true : ndarray of shape (n_samples,)
            Labels vrais
        y_pred : ndarray of shape (n_samples,)
            Labels prédits par clustering
            
        Returns
        -------
        metrics : Dict[str, float]
            Métriques de clustering (ARI, NMI, Silhouette)
        """
        metrics = {}
        
        try:
            # ARI (Adjusted Rand Index)
            metrics['ari_score'] = adjusted_rand_score(y_true, y_pred)
        except Exception:
            metrics['ari_score'] = 0.0
        
        try:
            # NMI (Normalized Mutual Information)
            metrics['nmi_score'] = normalized_mutual_info_score(
                y_true, y_pred, average_method='arithmetic'
            )
        except Exception:
            metrics['nmi_score'] = 0.0
        
        try:
            # Silhouette Score (cohésion interne des clusters)
            if len(np.unique(y_pred)) > 1:  # Au moins 2 clusters
                metrics['silhouette_score'] = silhouette_score(X_embedded, y_pred)
            else:
                metrics['silhouette_score'] = 0.0
        except Exception:
            metrics['silhouette_score'] = 0.0
        
        return metrics
    
    def calculate_local_metrics(self, X_high: np.ndarray, 
                               X_low: np.ndarray,
                               k_values: List[int] = [5, 10, 15, 20]) -> Dict[str, List[float]]:
        """
        Calcule les métriques locales pour différentes valeurs de k.
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)
            Embedding basse dimension
        k_values : List[int], default=[5, 10, 15, 20]
            Valeurs de k à tester
            
        Returns
        -------
        local_metrics : Dict[str, List[float]]
            Métriques pour chaque valeur de k
        """
        local_metrics = {
            'k_values': k_values,
            'trustworthiness': [],
            'continuity': []
        }
        
        max_k = X_high.shape[0] - 1
        
        for k in k_values:
            if k >= max_k:
                local_metrics['trustworthiness'].append(np.nan)
                local_metrics['continuity'].append(np.nan)
                continue
            
            try:
                trust = self.calculate_trustworthiness(X_high, X_low, k)
                cont = self.calculate_continuity(X_high, X_low, k)
                
                local_metrics['trustworthiness'].append(trust)
                local_metrics['continuity'].append(cont)
                
            except Exception as e:
                warnings.warn(f"Erreur métriques locales k={k}: {str(e)}")
                local_metrics['trustworthiness'].append(np.nan)
                local_metrics['continuity'].append(np.nan)
        
        return local_metrics
    
    def calculate_global_metrics(self, X_high: np.ndarray, 
                                X_low: np.ndarray) -> Dict[str, float]:
        """
        Calcule des métriques globales supplémentaires.
        
        Parameters
        ----------
        X_high : ndarray of shape (n_samples, n_features)
            Données haute dimension
        X_low : ndarray of shape (n_samples, n_components)  
            Embedding basse dimension
            
        Returns
        -------
        global_metrics : Dict[str, float]
            Métriques globales
        """
        global_metrics = {}
        
        try:
            # Corrélation de rang de Spearman
            n_samples = X_high.shape[0]
            sample_size = min(1000, n_samples)
            
            if n_samples > sample_size:
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_h_sample = X_high[indices]
                X_l_sample = X_low[indices]
            else:
                X_h_sample = X_high
                X_l_sample = X_low
            
            dist_high = pdist(X_h_sample)
            dist_low = pdist(X_l_sample)
            
            spearman_corr, _ = spearmanr(dist_high, dist_low)
            global_metrics['spearman_correlation'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            
        except Exception as e:
            warnings.warn(f"Erreur corrélation Spearman: {str(e)}")
            global_metrics['spearman_correlation'] = 0.0
        
        try:
            # Variance expliquée (approximation)
            var_high = np.var(X_high, axis=0).sum()
            var_low = np.var(X_low, axis=0).sum()
            
            global_metrics['variance_ratio'] = var_low / var_high if var_high > 0 else 0.0
            
        except Exception as e:
            warnings.warn(f"Erreur variance ratio: {str(e)}")
            global_metrics['variance_ratio'] = 0.0
        
        return global_metrics
    
    def create_metrics_summary(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Crée un résumé formaté des métriques.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Métriques calculées
            
        Returns
        -------
        summary : pd.DataFrame
            Résumé formaté des métriques
        """
        categories = {
            'Structure Locale': {
                'trustworthiness': 'Trustworthiness',
                'continuity': 'Continuity', 
                'knn_accuracy': 'KNN Accuracy'
            },
            'Corrélations': {
                'shepard_correlation': 'Shepard (Pearson)',
                'spearman_correlation': 'Spearman'
            },
            'Clustering': {
                'silhouette_score': 'Silhouette Score',
                'ari_score': 'Adjusted Rand Index',
                'nmi_score': 'Normalized MI'
            },
            'Distorsions': {
                'stress': 'Stress',
                'normalized_stress': 'Normalized Stress'
            }
        }
        
        summary_data = []
        
        for category, metric_names in categories.items():
            for metric_key, display_name in metric_names.items():
                if metric_key in metrics:
                    value = metrics[metric_key]
                    
                    if metric_key in ['stress', 'normalized_stress']:
                        formatted_value = f"{value:.4f}" if value < np.inf else "∞"
                        interpretation = "Plus faible = Meilleur"
                    else:
                        formatted_value = f"{value:.4f}"
                        interpretation = "Plus élevé = Meilleur"
                    
                    summary_data.append({
                        'Catégorie': category,
                        'Métrique': display_name,
                        'Valeur': formatted_value,
                        'Interprétation': interpretation
                    })
        
        return pd.DataFrame(summary_data)
    
    def compare_metrics(self, metrics1: Dict[str, float], 
                       metrics2: Dict[str, float],
                       names: Tuple[str, str] = ("Modèle 1", "Modèle 2")) -> pd.DataFrame:
        """
        Compare deux ensembles de métriques.
        
        Parameters
        ----------
        metrics1 : Dict[str, float]
            Métriques du premier modèle
        metrics2 : Dict[str, float]
            Métriques du second modèle
        names : Tuple[str, str], default=("Modèle 1", "Modèle 2")
            Noms des modèles
            
        Returns
        -------
        comparison : pd.DataFrame
            Tableau de comparaison
        """
        comparison_data = []
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        metric_display_names = {
            'trustworthiness': 'Trustworthiness',
            'continuity': 'Continuity',
            'shepard_correlation': 'Shepard Correlation',
            'knn_accuracy': 'KNN Accuracy',
            'silhouette_score': 'Silhouette Score',
            'ari_score': 'ARI',
            'nmi_score': 'NMI',
            'stress': 'Stress',
            'normalized_stress': 'Normalized Stress'
        }
        
        for metric_key in sorted(common_metrics):
            if metric_key in metric_display_names:
                display_name = metric_display_names[metric_key]
                
                val1 = metrics1[metric_key]
                val2 = metrics2[metric_key]
                
                if metric_key in ['stress', 'normalized_stress']:
                    diff = val1 - val2
                    winner = names[0] if val1 < val2 else names[1]
                    if val1 == val2:
                        winner = "Égalité"
                else:
                    diff = val1 - val2
                    winner = names[0] if val1 > val2 else names[1]
                    if val1 == val2:
                        winner = "Égalité"
                
                comparison_data.append({
                    'Métrique': display_name,
                    names[0]: f"{val1:.4f}",
                    names[1]: f"{val2:.4f}",
                    'Différence': f"{diff:+.4f}",
                    'Meilleur': winner
                })
        
        return pd.DataFrame(comparison_data)