"""
Gestionnaire de parameter sweeps pour UMAP vs t-SNE Explorer - Version Complète
==============================================================================

Implémente la recherche systématique d'hyperparamètres :
- Grilles de paramètres configurables pour UMAP et t-SNE
- Évaluation avec métriques multiples
- Génération automatique de heatmaps des résultats
- Gestion d'erreurs robuste
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import time
from itertools import product
import warnings

from .embeddings import EmbeddingManager
from .metrics import MetricsCalculator
from .plots import PlotManager
from .utils import seed_everything

class SweepManager:
    """Gestionnaire centralisé pour les parameter sweeps."""
    
    def __init__(self, n_jobs: int = 1):
        """
        Initialise le gestionnaire de sweeps.
        
        Parameters
        ----------
        n_jobs : int, default=1
            Nombre de threads pour parallélisation (1 = séquentiel)
        """
        seed_everything(42)
        self.n_jobs = n_jobs
        self.embedding_manager = EmbeddingManager()
        self.metrics_calculator = MetricsCalculator()
        self.plot_manager = PlotManager()
    
    def umap_parameter_sweep(self, X: np.ndarray,
                           y: np.ndarray,
                           evaluation_metric: str = 'trustworthiness',
                           param_grid: Optional[Dict[str, List]] = None) -> Tuple[Dict[str, List], plt.Figure, Dict[str, Any]]:
        """
        Effectue un parameter sweep pour UMAP.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Données preprocessées
        y : ndarray of shape (n_samples,)
            Labels vrais
        evaluation_metric : str, default='trustworthiness'
            Métrique d'évaluation principale
        param_grid : Dict[str, List], optional
            Grille de paramètres personnalisée

        Returns
        -------
        results : Dict[str, List]
            Résultats détaillés du sweep
        heatmap_fig : plt.Figure
            Heatmap des résultats
        best_params : Dict[str, Any]
            Meilleurs hyperparamètres trouvés
        """
        # Grille par défaut si non spécifiée
        if param_grid is None:
            param_grid = {
                'n_neighbors': [5, 10, 15, 30, 50, 100],
                'min_dist': [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
            }
        
        # Préparation des combinaisons
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"🔍 UMAP Parameter Sweep: {len(param_combinations)} combinaisons à tester")
        
        # Stockage des résultats
        results = {name: [] for name in param_names}
        results.update({
            evaluation_metric: [],
            'trustworthiness': [],
            'continuity': [],
            'knn_accuracy': [],
            'execution_time': []
        })
        
        # Exécution séquentielle
        for i, params in enumerate(param_combinations):
            result = self._evaluate_umap_params(X, y, param_names, params, evaluation_metric)
            self._store_result(results, param_names, params, result)
            
            # Progression
            if (i + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"Progression: {i + 1}/{len(param_combinations)} ({100 * (i + 1) / len(param_combinations):.1f}%)")
        
        # Création de la heatmap
        heatmap_fig = self._create_2d_heatmap(
            results, param_names, param_grid, evaluation_metric, 'UMAP'
        )
        
        # Trouver les meilleurs paramètres
        best_params = {}
        if len(results[evaluation_metric]) > 0:
            best_idx = np.argmax(results[evaluation_metric])
            best_score = results[evaluation_metric][best_idx]

            # Construire le dictionnaire des meilleurs paramètres
            for param in param_names:
                best_params[param] = results[param][best_idx]

            # Ajouter les paramètres par défaut UMAP (sans écraser ceux du sweep)
            defaults = {
                'metric': 'euclidean',
                'init': 'spectral',
                'learning_rate': 1.0,
                'n_components': 2,
                'random_state': 42
            }
            for key, value in defaults.items():
                if key not in best_params:
                    best_params[key] = value

            print(f"✅ UMAP Sweep terminé. Meilleur {evaluation_metric}: {best_score:.4f}")
            print(f"   Meilleurs paramètres: {best_params}")

        return results, heatmap_fig, best_params
    
    def tsne_parameter_sweep(self, X: np.ndarray,
                           y: np.ndarray,
                           evaluation_metric: str = 'trustworthiness',
                           param_grid: Optional[Dict[str, List]] = None) -> Tuple[Dict[str, List], plt.Figure, Dict[str, Any]]:
        """
        Effectue un parameter sweep pour t-SNE.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Données preprocessées
        y : ndarray of shape (n_samples,)
            Labels vrais
        evaluation_metric : str, default='trustworthiness'
            Métrique d'évaluation principale
        param_grid : Dict[str, List], optional
            Grille de paramètres personnalisée
            
        Returns
        -------
        results : Dict[str, List]
            Résultats détaillés du sweep
        heatmap_fig : plt.Figure
            Heatmap des résultats
        best_params : Dict[str, Any]
            Meilleurs hyperparamètres trouvés
        """
        # Grille par défaut
        if param_grid is None:
            max_perplexity = min(100, (X.shape[0] - 1) // 3)
            param_grid = {
                'perplexity': [5, 10, 20, 30, 50, min(80, max_perplexity)],
                'learning_rate': ['auto', 10, 50, 200, 1000]
            }
        
        # Validation de perplexity
        max_perplexity = (X.shape[0] - 1) // 3
        if 'perplexity' in param_grid:
            param_grid['perplexity'] = [p for p in param_grid['perplexity'] if p < max_perplexity]
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"🔍 t-SNE Parameter Sweep: {len(param_combinations)} combinaisons à tester")
        
        # Stockage des résultats
        results = {name: [] for name in param_names}
        results.update({
            evaluation_metric: [],
            'trustworthiness': [],
            'continuity': [], 
            'knn_accuracy': [],
            'execution_time': []
        })
        
        # Exécution
        for i, params in enumerate(param_combinations):
            result = self._evaluate_tsne_params(X, y, param_names, params, evaluation_metric)
            self._store_result(results, param_names, params, result)
            
            # Progression
            if (i + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"Progression: {i + 1}/{len(param_combinations)} ({100 * (i + 1) / len(param_combinations):.1f}%)")
        
        # Création de la heatmap
        heatmap_fig = self._create_2d_heatmap(
            results, param_names, param_grid, evaluation_metric, 't-SNE'
        )
        
        # Trouver les meilleurs paramètres
        best_params = {}
        if len(results[evaluation_metric]) > 0:
            best_idx = np.argmax(results[evaluation_metric])
            best_score = results[evaluation_metric][best_idx]

            # Construire le dictionnaire des meilleurs paramètres
            for param in param_names:
                best_params[param] = results[param][best_idx]

            # Ajouter les paramètres par défaut t-SNE
            best_params.update({
                'n_components': 2,
                'init': 'pca',
                'random_state': 42,
                'pca_prenorm': True
            })

            print(f"✅ t-SNE Sweep terminé. Meilleur {evaluation_metric}: {best_score:.4f}")
            print(f"   Meilleurs paramètres: {best_params}")

        return results, heatmap_fig, best_params
    
    def _evaluate_umap_params(self, X: np.ndarray, 
                             y: np.ndarray,
                             param_names: List[str],
                             param_values: Tuple,
                             evaluation_metric: str) -> Dict[str, float]:
        """Évalue une combinaison de paramètres UMAP."""
        try:
            # Construction des paramètres
            params = dict(zip(param_names, param_values))
            
            # Calcul de l'embedding
            start_time = time.time()
            X_embedded, _ = self.embedding_manager.compute_umap(X, params)
            execution_time = time.time() - start_time
            
            # Calcul des métriques
            metrics = self.metrics_calculator.calculate_all_metrics(X, X_embedded, y)
            
            result = {
                'trustworthiness': metrics.get('trustworthiness', 0.0),
                'continuity': metrics.get('continuity', 0.0),
                'knn_accuracy': metrics.get('knn_accuracy', 0.0),
                'execution_time': execution_time
            }
            result[evaluation_metric] = metrics.get(evaluation_metric, 0.0)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Erreur évaluation UMAP {params}: {str(e)}")
            return self._empty_evaluation_result(evaluation_metric)
    
    def _evaluate_tsne_params(self, X: np.ndarray,
                             y: np.ndarray,
                             param_names: List[str], 
                             param_values: Tuple,
                             evaluation_metric: str) -> Dict[str, float]:
        """Évalue une combinaison de paramètres t-SNE."""
        try:
            # Construction des paramètres
            params = dict(zip(param_names, param_values))
            
            # Calcul de l'embedding
            start_time = time.time()
            X_embedded, _ = self.embedding_manager.compute_tsne(X, params)
            execution_time = time.time() - start_time
            
            # Calcul des métriques
            metrics = self.metrics_calculator.calculate_all_metrics(X, X_embedded, y)
            
            result = {
                'trustworthiness': metrics.get('trustworthiness', 0.0),
                'continuity': metrics.get('continuity', 0.0),
                'knn_accuracy': metrics.get('knn_accuracy', 0.0),
                'execution_time': execution_time
            }
            result[evaluation_metric] = metrics.get(evaluation_metric, 0.0)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Erreur évaluation t-SNE {params}: {str(e)}")
            return self._empty_evaluation_result(evaluation_metric)
    
    def _store_result(self, results: Dict[str, List], param_names: List[str],
                     param_values: Tuple, result: Dict[str, float]):
        """Stocke un résultat d'évaluation."""
        # Paramètres
        for name, value in zip(param_names, param_values):
            results[name].append(value)
        
        # Métriques
        for metric_name, metric_value in result.items():
            if metric_name in results:
                results[metric_name].append(metric_value)
    
    def _empty_evaluation_result(self, evaluation_metric: str) -> Dict[str, float]:
        """Retourne un résultat d'évaluation vide."""
        result = {
            'trustworthiness': 0.0,
            'continuity': 0.0,
            'knn_accuracy': 0.0,
            'execution_time': np.inf
        }
        result[evaluation_metric] = 0.0
        return result
    
    def _create_2d_heatmap(self, results: Dict[str, List],
                          param_names: List[str],
                          param_grid: Dict[str, List],
                          evaluation_metric: str,
                          algorithm: str) -> plt.Figure:
        """
        Crée une heatmap 2D des résultats de sweep.
        
        Parameters
        ----------
        results : Dict[str, List]
            Résultats du sweep
        param_names : List[str]
            Noms des paramètres
        param_grid : Dict[str, List]
            Grille des paramètres
        evaluation_metric : str
            Métrique d'évaluation
        algorithm : str
            Nom de l'algorithme
            
        Returns
        -------
        fig : plt.Figure
            Heatmap des résultats
        """
        if len(param_names) != 2:
            # Fallback pour plus de 2 paramètres : graphique en barres
            return self._create_bar_plot_results(results, param_names, evaluation_metric, algorithm)
        
        # Reconstruction de la grille 2D
        param1_name, param2_name = param_names
        param1_values = param_grid[param1_name]
        param2_values = param_grid[param2_name]
        
        # Matrice des résultats
        results_matrix = np.full((len(param1_values), len(param2_values)), np.nan)
        
        for i, (p1, p2, metric_val) in enumerate(zip(results[param1_name], 
                                                    results[param2_name],
                                                    results[evaluation_metric])):
            try:
                idx1 = param1_values.index(p1)
                idx2 = param2_values.index(p2)
                results_matrix[idx1, idx2] = metric_val
            except ValueError:
                continue
        
        # Création de la heatmap
        fig = self.plot_manager.plot_metrics_heatmap(
            results_matrix, param1_values, param2_values,
            param1_name, param2_name, f'{algorithm} - {evaluation_metric}'
        )
        
        return fig
    
    def _create_bar_plot_results(self, results: Dict[str, List],
                                param_names: List[str],
                                evaluation_metric: str,
                                algorithm: str) -> plt.Figure:
        """Crée un graphique en barres pour plus de 2 paramètres."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Création des labels pour chaque combinaison
        n_results = len(results[evaluation_metric])
        labels = []
        
        for i in range(n_results):
            param_str = ", ".join([f"{name}={results[name][i]}" for name in param_names])
            labels.append(param_str)
        
        # Limitation à 20 meilleurs résultats pour lisibilité
        if n_results > 20:
            # Tri par métrique d'évaluation
            sorted_indices = np.argsort(results[evaluation_metric])[-20:]
            
            plot_values = [results[evaluation_metric][i] for i in sorted_indices]
            plot_labels = [labels[i] for i in sorted_indices]
        else:
            plot_values = results[evaluation_metric]
            plot_labels = labels
        
        # Graphique en barres
        bars = ax.barh(range(len(plot_values)), plot_values, alpha=0.8, color='steelblue')
        
        ax.set_yticks(range(len(plot_labels)))
        ax.set_yticklabels(plot_labels, fontsize=8)
        ax.set_xlabel(f'{evaluation_metric}', fontsize=12)
        ax.set_title(f'{algorithm} Parameter Sweep - Top Résultats', fontsize=14, fontweight='bold')
        
        # Annotation des valeurs
        for i, (bar, val) in enumerate(zip(bars, plot_values)):
            width = bar.get_width()
            ax.text(width + max(plot_values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_comparison_sweep(self, X: np.ndarray, y: np.ndarray,
                               evaluation_metrics: List[str] = ['trustworthiness', 'knn_accuracy'],
                               umap_grid: Optional[Dict[str, List]] = None,
                               tsne_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Compare UMAP et t-SNE sur plusieurs métriques avec parameter sweep.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Données preprocessées
        y : ndarray of shape (n_samples,)
            Labels vrais
        evaluation_metrics : List[str], default=['trustworthiness', 'knn_accuracy']
            Métriques à comparer
        umap_grid : Dict[str, List], optional
            Grille UMAP personnalisée
        tsne_grid : Dict[str, List], optional
            Grille t-SNE personnalisée
            
        Returns
        -------
        comparison_results : Dict[str, Any]
            Résultats de comparaison complète
        """
        print("🔍 Démarrage du sweep comparatif UMAP vs t-SNE...")
        
        comparison_results = {
            'umap_results': {},
            'tsne_results': {},
            'comparison_summary': {},
            'figures': {}
        }
        
        # Sweep UMAP pour chaque métrique
        for metric in evaluation_metrics:
            print(f"\n📊 UMAP sweep pour {metric}...")
            umap_results, umap_fig = self.umap_parameter_sweep(
                X, y, evaluation_metric=metric, param_grid=umap_grid
            )
            comparison_results['umap_results'][metric] = umap_results
            comparison_results['figures'][f'umap_{metric}_heatmap'] = umap_fig
        
        # Sweep t-SNE pour chaque métrique
        for metric in evaluation_metrics:
            print(f"\n📊 t-SNE sweep pour {metric}...")
            tsne_results, tsne_fig = self.tsne_parameter_sweep(
                X, y, evaluation_metric=metric, param_grid=tsne_grid
            )
            comparison_results['tsne_results'][metric] = tsne_results
            comparison_results['figures'][f'tsne_{metric}_heatmap'] = tsne_fig
        
        # Résumé comparatif
        summary = self._create_sweep_summary(
            comparison_results['umap_results'],
            comparison_results['tsne_results'],
            evaluation_metrics
        )
        comparison_results['comparison_summary'] = summary
        
        # Figure de résumé
        summary_fig = self._create_sweep_comparison_plot(summary, evaluation_metrics)
        comparison_results['figures']['comparison_summary'] = summary_fig
        
        print("\n✅ Sweep comparatif terminé !")
        
        return comparison_results
    
    def _create_sweep_summary(self, umap_results: Dict[str, Dict[str, List]],
                             tsne_results: Dict[str, Dict[str, List]], 
                             evaluation_metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Crée un résumé comparatif des sweeps."""
        summary = {}
        
        for metric in evaluation_metrics:
            if metric in umap_results and metric in tsne_results:
                umap_values = umap_results[metric][metric]
                tsne_values = tsne_results[metric][metric]
                
                summary[metric] = {
                    'umap_best': max(umap_values) if umap_values else 0.0,
                    'umap_mean': np.mean(umap_values) if umap_values else 0.0,
                    'umap_std': np.std(umap_values) if umap_values else 0.0,
                    'tsne_best': max(tsne_values) if tsne_values else 0.0,
                    'tsne_mean': np.mean(tsne_values) if tsne_values else 0.0,
                    'tsne_std': np.std(tsne_values) if tsne_values else 0.0,
                    'winner': 'UMAP' if max(umap_values) > max(tsne_values) else 't-SNE'
                }
        
        return summary
    
    def _create_sweep_comparison_plot(self, summary: Dict[str, Dict[str, float]],
                                     evaluation_metrics: List[str]) -> plt.Figure:
        """Crée un graphique de comparaison des sweeps."""
        n_metrics = len(evaluation_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), dpi=100)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(evaluation_metrics):
            if metric not in summary:
                continue
                
            ax = axes[i] if n_metrics > 1 else axes[0]
            metric_data = summary[metric]
            
            # Valeurs moyennes avec barres d'erreur
            algorithms = ['UMAP', 't-SNE']
            means = [metric_data['umap_mean'], metric_data['tsne_mean']]
            stds = [metric_data['umap_std'], metric_data['tsne_std']]
            
            bars = ax.bar(algorithms, means, yerr=stds, capsize=5,
                         color=['#2E86AB', '#A23B72'], alpha=0.8, error_kw={'linewidth': 2})
            
            # Marquage des valeurs maximales
            max_values = [metric_data['umap_best'], metric_data['tsne_best']]
            for j, (bar, mean_val, max_val) in enumerate(zip(bars, means, max_values)):
                # Valeur moyenne
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[j] + 0.01,
                       f'μ={mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Valeur maximale
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[j] + 0.05,
                       f'max={max_val:.3f}', ha='center', va='bottom', 
                       fontsize=9, style='italic')
            
            ax.set_title(f'{metric.title()}\n(Gagnant: {metric_data["winner"]})', 
                        fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Comparaison Parameter Sweeps: UMAP vs t-SNE', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def export_sweep_results(self, results: Dict[str, List],
                           filename: str = "sweep_results.csv") -> str:
        """
        Exporte les résultats d'un sweep vers un fichier CSV.
        
        Parameters
        ----------
        results : Dict[str, List]
            Résultats du sweep
        filename : str, default="sweep_results.csv"
            Nom du fichier de sortie
            
        Returns
        -------
        filepath : str
            Chemin du fichier créé
        """
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        
        print(f"💾 Résultats exportés vers: {filename}")
        return filename