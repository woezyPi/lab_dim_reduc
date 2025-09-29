"""
Gestionnaire de visualisations pour UMAP vs t-SNE Explorer - Version Complète
=============================================================================

Gère la création de tous les graphiques :
- Scatter plots d'embeddings (Matplotlib + Plotly)
- Bar charts comparatifs harmonisés
- Heatmaps pour parameter sweeps
- Palettes de couleurs cohérentes entre algorithmes
- Légendes discrètes pour classes nominales
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

from .utils import seed_everything

class PlotManager:
    """Gestionnaire centralisé pour toutes les visualisations."""
    
    def __init__(self, use_plotly: bool = False, style: str = 'default'):
        """
        Initialise le gestionnaire de plots.
        
        Parameters
        ----------
        use_plotly : bool, default=False
            Utiliser Plotly au lieu de Matplotlib
        style : str, default='default'
            Style matplotlib à utiliser
        """
        seed_everything(42)
        self.use_plotly = use_plotly
        
        # Configuration matplotlib
        if not use_plotly:
            try:
                plt.style.use(style if style in plt.style.available else 'default')
            except:
                plt.style.use('default')
            
        # Palette de couleurs fixe et cohérente
        self.color_palette = self._create_color_palette()
    
    def _create_color_palette(self) -> List[str]:
        """
        Crée une palette de couleurs distinctes et cohérentes.
        
        Returns
        -------
        palette : List[str]
            Liste de couleurs hexadécimales
        """
        # Palette optimisée pour distinguabilité et esthétique
        base_colors = [
            '#1f77b4',  # Bleu
            '#ff7f0e',  # Orange
            '#2ca02c',  # Vert
            '#d62728',  # Rouge
            '#9467bd',  # Violet
            '#8c564b',  # Marron
            '#e377c2',  # Rose
            '#7f7f7f',  # Gris
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#aec7e8',  # Bleu clair
            '#ffbb78',  # Orange clair
            '#98df8a',  # Vert clair
            '#ff9896',  # Rouge clair
            '#c5b0d5',  # Violet clair
            '#c49c94',  # Marron clair
            '#f7b6d3',  # Rose clair
            '#c7c7c7',  # Gris clair
            '#dbdb8d',  # Olive clair
            '#9edae5'   # Cyan clair
        ]
        
        return base_colors
    
    def plot_embedding(self, X_embedded: np.ndarray, 
                      y: np.ndarray,
                      class_names: List[str],
                      title: str,
                      point_size: int = 10,
                      alpha: float = 0.7,
                      figsize: Tuple[int, int] = (8, 6)) -> Union[plt.Figure, go.Figure]:
        """
        Crée un scatter plot de l'embedding avec légende discrète.
        
        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, 2)
            Coordonnées 2D de l'embedding
        y : ndarray of shape (n_samples,)
            Labels des classes
        class_names : List[str]
            Noms des classes
        title : str
            Titre du graphique
        point_size : int, default=10
            Taille des points
        alpha : float, default=0.7
            Transparence des points
        figsize : Tuple[int, int], default=(8, 6)
            Taille de la figure
            
        Returns
        -------
        fig : matplotlib.Figure or plotly.Figure
            Figure créée
        """
        if self.use_plotly:
            return self._plot_embedding_plotly(
                X_embedded, y, class_names, title, point_size, alpha
            )
        else:
            return self._plot_embedding_matplotlib(
                X_embedded, y, class_names, title, point_size, alpha, figsize
            )
    
    def _plot_embedding_matplotlib(self, X_embedded: np.ndarray,
                                  y: np.ndarray,
                                  class_names: List[str], 
                                  title: str,
                                  point_size: int,
                                  alpha: float,
                                  figsize: Tuple[int, int]) -> plt.Figure:
        """Crée un scatter plot avec Matplotlib."""
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Attribution des couleurs cohérentes
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        colors = self.color_palette[:n_classes]
        
        # Plot par classe pour avoir une légende propre
        for i, label in enumerate(unique_labels):
            mask = y == label
            class_name = class_names[label] if label < len(class_names) else f'Classe {label}'
            
            ax.scatter(
                X_embedded[mask, 0], 
                X_embedded[mask, 1],
                s=point_size,
                alpha=alpha,
                color=colors[i],
                label=class_name,
                edgecolors='white',
                linewidth=0.1
            )
        
        # Stylisation
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        
        # Légende adaptative
        if n_classes <= 15:  # Légende visible si pas trop de classes
            legend = ax.legend(
                bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True,
                ncol=1 if n_classes <= 10 else 2,
                fontsize=10
            )
            legend.get_frame().set_alpha(0.9)
        
        # Grille subtile
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Égalisation des axes pour éviter la distorsion
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        return fig
    
    def _plot_embedding_plotly(self, X_embedded: np.ndarray,
                              y: np.ndarray,
                              class_names: List[str],
                              title: str,
                              point_size: int,
                              alpha: float) -> go.Figure:
        """Crée un scatter plot avec Plotly."""
        # Préparation des données
        df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            'class_idx': y,
            'class_name': [class_names[label] if label < len(class_names) else f'Classe {label}' for label in y]
        })
        
        # Création du graphique
        fig = px.scatter(
            df, x='x', y='y',
            color='class_name',
            color_discrete_sequence=self.color_palette,
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            hover_data={'class_idx': False, 'class_name': True}
        )
        
        # Stylisation
        fig.update_traces(
            marker=dict(
                size=point_size,
                opacity=alpha,
                line=dict(width=0.1, color='white')
            )
        )
        
        fig.update_layout(
            title=dict(font=dict(size=16)),
            xaxis_title=dict(font=dict(size=12)),
            yaxis_title=dict(font=dict(size=12)),
            legend=dict(
                title="Classes",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=600
        )
        
        # Grille
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def plot_comparison_bar(self, algorithms: List[str],
                           values: List[float],
                           metric_name: str,
                           figsize: Tuple[int, int] = (6, 4)) -> plt.Figure:
        """
        Crée un graphique en barres pour comparer les algorithmes.
        
        Parameters
        ----------
        algorithms : List[str]
            Noms des algorithmes
        values : List[float]
            Valeurs de la métrique
        metric_name : str
            Nom de la métrique
        figsize : Tuple[int, int], default=(6, 4)
            Taille de la figure
            
        Returns
        -------
        fig : matplotlib.Figure
            Figure créée
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Couleurs distinctes
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(algorithms)]
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.8, 
                     edgecolor='white', linewidth=1)
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Stylisation
        ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel('Valeur', fontsize=10)
        
        # Harmonisation des axes Y
        y_min = min(0, min(values) * 0.95)
        y_max = max(values) * 1.15
        ax.set_ylim(y_min, y_max)
        
        # Grille horizontale
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Suppression des bordures supérieure et droite
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_heatmap(self, results_grid: np.ndarray,
                            param1_values: List,
                            param2_values: List,
                            param1_name: str,
                            param2_name: str,
                            metric_name: str,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Crée une heatmap pour visualiser les résultats de parameter sweep.
        
        Parameters
        ----------
        results_grid : ndarray of shape (n_param1, n_param2)
            Grille des résultats
        param1_values : List
            Valeurs du premier paramètre (axe Y)
        param2_values : List
            Valeurs du second paramètre (axe X)
        param1_name : str
            Nom du premier paramètre
        param2_name : str
            Nom du second paramètre
        metric_name : str
            Nom de la métrique
        figsize : Tuple[int, int], default=(10, 8)
            Taille de la figure
            
        Returns
        -------
        fig : matplotlib.Figure
            Figure créée
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Choix de la colormap selon la métrique
        if 'stress' in metric_name.lower():
            cmap = 'viridis_r'  # Inversé car plus bas = meilleur
        else:
            cmap = 'viridis'
        
        # Heatmap
        im = ax.imshow(results_grid, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Configuration des ticks
        ax.set_xticks(range(len(param2_values)))
        ax.set_yticks(range(len(param1_values)))
        ax.set_xticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) 
                           for v in param2_values])
        ax.set_yticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) 
                           for v in param1_values])
        
        # Labels
        ax.set_xlabel(param2_name, fontsize=12)
        ax.set_ylabel(param1_name, fontsize=12)
        ax.set_title(f'Parameter Sweep: {metric_name}', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_name, fontsize=11)
        
        # Annotation des valeurs sur les cellules (si pas trop nombreuses)
        if results_grid.size <= 64:  # Max 8x8
            for i in range(len(param1_values)):
                for j in range(len(param2_values)):
                    value = results_grid[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < np.nanmean(results_grid) else 'black'
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color=text_color, fontsize=8, fontweight='bold')
        
        # Marquage du meilleur résultat
        if not np.all(np.isnan(results_grid)):
            if 'stress' in metric_name.lower():
                best_idx = np.unravel_index(np.nanargmin(results_grid), results_grid.shape)
            else:
                best_idx = np.unravel_index(np.nanargmax(results_grid), results_grid.shape)
            
            # Cercle autour du meilleur résultat
            circle = plt.Circle((best_idx[1], best_idx[0]), 0.3, 
                              fill=False, color='red', linewidth=3)
            ax.add_patch(circle)
        
        plt.tight_layout()
        
        return fig
    
    def plot_cluster_overlay(self, X_embedded: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            algorithm_name: str,
                            figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        """
        Affiche côte à côte les vraies classes et les clusters prédits.
        
        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, 2)
            Coordonnées 2D de l'embedding
        y_true : ndarray of shape (n_samples,)
            Vraies classes
        y_pred : ndarray of shape (n_samples,)
            Clusters prédits
        class_names : List[str]
            Noms des vraies classes
        algorithm_name : str
            Nom de l'algorithme de clustering
        figsize : Tuple[int, int], default=(16, 6)
            Taille de la figure
            
        Returns
        -------
        fig : matplotlib.Figure
            Figure avec deux sous-graphiques
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
        
        # Graphique 1: Vraies classes
        unique_true = np.unique(y_true)
        for i, label in enumerate(unique_true):
            mask = y_true == label
            class_name = class_names[label] if label < len(class_names) else f'Classe {label}'
            ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                       s=20, alpha=0.7, color=self.color_palette[i],
                       label=class_name, edgecolors='white', linewidth=0.1)
        
        ax1.set_title('Vraies Classes', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Graphique 2: Clusters prédits
        unique_pred = np.unique(y_pred)
        cluster_colors = self.color_palette[:len(unique_pred)]
        
        for i, label in enumerate(unique_pred):
            mask = y_pred == label
            if label == -1:  # Points de bruit
                ax2.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                           s=20, alpha=0.5, color='gray', marker='x',
                           label='Bruit')
            else:
                ax2.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                           s=20, alpha=0.7, color=cluster_colors[i],
                           label=f'Cluster {label}', edgecolors='white', linewidth=0.1)
        
        ax2.set_title(f'Clusters {algorithm_name}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Égalisation des axes
        for ax in [ax1, ax2]:
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        return fig
    
    def create_summary_dashboard(self, umap_metrics: Dict[str, float],
                               tsne_metrics: Dict[str, float],
                               umap_time: float,
                               tsne_time: float) -> plt.Figure:
        """
        Crée un dashboard résumé avec toutes les métriques importantes.
        
        Parameters
        ----------
        umap_metrics : Dict[str, float]
            Métriques UMAP
        tsne_metrics : Dict[str, float]
            Métriques t-SNE
        umap_time : float
            Temps d'exécution UMAP
        tsne_time : float
            Temps d'exécution t-SNE
            
        Returns
        -------
        fig : matplotlib.Figure
            Dashboard complet
        """
        fig = plt.figure(figsize=(14, 8), dpi=100)
        
        # Grille de sous-graphiques
        gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)
        
        # Métriques principales
        main_metrics = [
            ('Trustworthiness', 'trustworthiness'),
            ('Continuity', 'continuity'), 
            ('KNN Accuracy', 'knn_accuracy'),
            ('Shepard Correlation', 'shepard_correlation')
        ]
        
        for i, (name, key) in enumerate(main_metrics):
            ax = fig.add_subplot(gs[0, i])
            
            values = [umap_metrics.get(key, 0), tsne_metrics.get(key, 0)]
            bars = ax.bar(['UMAP', 't-SNE'], values, 
                         color=['#2E86AB', '#A23B72'], alpha=0.8)
            
            # Annotation des valeurs
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(name, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis='y', alpha=0.3)
        
        # Temps d'exécution
        ax_time = fig.add_subplot(gs[1, 0])
        times = [umap_time, tsne_time]
        bars_time = ax_time.bar(['UMAP', 't-SNE'], times, 
                               color=['#2E86AB', '#A23B72'], alpha=0.8)
        
        for bar, time_val in zip(bars_time, times):
            height = bar.get_height()
            ax_time.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                        f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax_time.set_title('Temps d\'Exécution', fontweight='bold')
        ax_time.set_ylabel('Temps (s)')
        ax_time.grid(True, axis='y', alpha=0.3)
        
        # Métriques de clustering
        cluster_metrics = [
            ('Silhouette Score', 'silhouette_score'),
            ('ARI', 'ari_score'),
            ('NMI', 'nmi_score')
        ]
        
        for i, (name, key) in enumerate(cluster_metrics):
            ax = fig.add_subplot(gs[1, i+1])
            
            values = [umap_metrics.get(key, 0), tsne_metrics.get(key, 0)]
            bars = ax.bar(['UMAP', 't-SNE'], values,
                         color=['#2E86AB', '#A23B72'], alpha=0.8)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(name, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Dashboard Comparatif UMAP vs t-SNE', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        return fig
    
    def save_plots(self, figures: Dict[str, plt.Figure], 
                   output_dir: str = "plots",
                   formats: List[str] = ['png']) -> Dict[str, str]:
        """
        Sauvegarde les figures dans différents formats.
        
        Parameters
        ----------
        figures : Dict[str, plt.Figure]
            Dictionnaire {nom: figure}
        output_dir : str, default="plots"
            Répertoire de sortie
        formats : List[str], default=['png']
            Formats de sauvegarde ('png', 'pdf', 'svg')
            
        Returns
        -------
        saved_files : Dict[str, str]
            Dictionnaire {nom_figure: chemin_fichier}
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for fig_name, fig in figures.items():
            for fmt in formats:
                filename = f"{fig_name}.{fmt}"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    if fmt == 'pdf':
                        fig.savefig(filepath, format='pdf', bbox_inches='tight', 
                                  dpi=300, facecolor='white')
                    elif fmt == 'svg':
                        fig.savefig(filepath, format='svg', bbox_inches='tight',
                                  facecolor='white')
                    else:  # PNG par défaut
                        fig.savefig(filepath, format='png', bbox_inches='tight',
                                  dpi=300, facecolor='white')
                    
                    saved_files[f"{fig_name}_{fmt}"] = filepath
                    
                except Exception as e:
                    warnings.warn(f"Erreur sauvegarde {fig_name}.{fmt}: {str(e)}")
        
        return saved_files