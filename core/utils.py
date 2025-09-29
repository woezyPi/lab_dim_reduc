"""
Utilitaires pour UMAP vs t-SNE Explorer - Version Complète
=========================================================

Fonctions utilitaires communes :
- Gestion des seeds pour reproductibilité
- Validation des paramètres
- Gestion de l'interface utilisateur Streamlit
- Export/Import de sessions
- Palettes de couleurs et formatage
"""

import numpy as np
import pandas as pd
import streamlit as st
import random
import os
import json
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from pathlib import Path

def seed_everything(seed: int = 42):
    """
    Configure tous les générateurs de nombres aléatoires pour la reproductibilité.
    
    Parameters
    ----------
    seed : int, default=42
        Graine aléatoire à utiliser
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def validate_tsne_parameters(params: Dict[str, Any], n_samples: int) -> List[str]:
    """
    Valide les paramètres t-SNE et retourne les erreurs.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Paramètres t-SNE à valider
    n_samples : int
        Nombre d'échantillons
        
    Returns
    -------
    errors : List[str]
        Liste des erreurs de validation
    """
    errors = []
    
    # Validation de perplexity
    perplexity = params.get('perplexity', 30)
    max_perplexity = (n_samples - 1) // 3
    
    if perplexity >= max_perplexity:
        errors.append(
            f"⚠️ perplexity ({perplexity}) trop élevée pour n_samples={n_samples}. "
            f"Maximum recommandé: {max_perplexity-1}"
        )
    
    if perplexity < 5:
        errors.append("⚠️ perplexity < 5 peut donner des résultats instables")
    
    # Validation de learning_rate
    learning_rate = params.get('learning_rate', 'auto')
    if learning_rate != 'auto' and isinstance(learning_rate, (int, float)):
        if learning_rate < 1:
            errors.append("⚠️ learning_rate très faible peut ralentir la convergence")
        elif learning_rate > 1000:
            errors.append("⚠️ learning_rate très élevé peut causer de l'instabilité")
    
    # Validation de max_iter
    max_iter = params.get('max_iter', 1000)
    if max_iter < 250:
        errors.append("⚠️ max_iter < 250 peut être insuffisant pour la convergence")
    
    return errors

def validate_umap_parameters(params: Dict[str, Any], n_samples: int) -> List[str]:
    """
    Valide les paramètres UMAP et retourne les avertissements.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Paramètres UMAP à valider
    n_samples : int
        Nombre d'échantillons
        
    Returns
    -------
    warnings : List[str]
        Liste des avertissements
    """
    warnings_list = []
    
    # Validation de n_neighbors
    n_neighbors = params.get('n_neighbors', 15)
    if n_neighbors >= n_samples:
        warnings_list.append(
            f"⚠️ n_neighbors ({n_neighbors}) >= n_samples ({n_samples}). "
            f"Maximum: {n_samples - 1}"
        )
    elif n_neighbors > n_samples // 2:
        warnings_list.append(
            f"⚠️ n_neighbors très élevé ({n_neighbors}) peut sur-lisser la structure locale"
        )
    elif n_neighbors < 2:
        warnings_list.append("⚠️ n_neighbors < 2 peut causer des erreurs")
    
    # Validation de min_dist vs spread
    min_dist = params.get('min_dist', 0.1)
    spread = params.get('spread', 1.0)
    
    if min_dist > spread:
        warnings_list.append(
            f"⚠️ min_dist ({min_dist}) > spread ({spread}) peut causer des problèmes"
        )
    
    if min_dist < 0:
        warnings_list.append("⚠️ min_dist < 0 n'est pas recommandé")
    
    # Validation de n_epochs
    n_epochs = params.get('n_epochs', 200)
    if n_epochs < 50:
        warnings_list.append("⚠️ n_epochs < 50 peut être insuffisant")
    elif n_epochs > 1000 and n_samples < 5000:
        warnings_list.append("⚠️ n_epochs élevé peut être inutile pour des petits datasets")
    
    return warnings_list

class UIUtils:
    """Utilitaires pour l'interface utilisateur Streamlit."""
    
    @staticmethod
    def display_metrics_card(algorithm_name: str, 
                           metrics: Dict[str, float],
                           clustering_results: Dict[str, Any]):
        """
        Affiche une carte de métriques dans Streamlit.
        
        Parameters
        ----------
        algorithm_name : str
            Nom de l'algorithme
        metrics : Dict[str, float]
            Métriques à afficher
        clustering_results : Dict[str, Any]
            Résultats de clustering
        """
        with st.container():
            st.markdown(f"**📊 Métriques {algorithm_name}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Trustworthiness", f"{metrics.get('trustworthiness', 0):.4f}")
                st.metric("KNN Accuracy", f"{metrics.get('knn_accuracy', 0):.4f}")
                st.metric("Silhouette Score", f"{metrics.get('silhouette_score', 0):.4f}")
            
            with col2:
                st.metric("Continuity", f"{metrics.get('continuity', 0):.4f}")
                st.metric("Shepard Correlation", f"{metrics.get('shepard_correlation', 0):.4f}")
                
                # Info clustering
                if clustering_results:
                    best_algo = clustering_results.get('best_algorithm', 'unknown')
                    n_clusters = clustering_results.get('n_clusters', 0)
                    st.metric("Clustering", f"{best_algo.upper()} ({n_clusters} clusters)")

        return None  # Éviter tout retour de DeltaGenerator
    
    @staticmethod
    def validate_parameters(params: Dict[str, Dict[str, Any]], n_samples: int) -> List[str]:
        """
        Valide tous les paramètres et retourne les avertissements.
        
        Parameters
        ----------
        params : Dict[str, Dict[str, Any]]
            Paramètres UMAP et t-SNE
        n_samples : int
            Nombre d'échantillons
            
        Returns
        -------
        warnings : List[str]
            Liste de tous les avertissements
        """
        all_warnings = []
        
        if 'umap' in params:
            umap_warnings = validate_umap_parameters(params['umap'], n_samples)
            all_warnings.extend([f"UMAP: {w}" for w in umap_warnings])
        
        if 'tsne' in params:
            tsne_warnings = validate_tsne_parameters(params['tsne'], n_samples)
            all_warnings.extend([f"t-SNE: {w}" for w in tsne_warnings])
        
        return all_warnings
    
    @staticmethod
    def create_comparison_table(umap_metrics: Dict[str, float],
                              tsne_metrics: Dict[str, float], 
                              umap_clustering: Dict[str, Any],
                              tsne_clustering: Dict[str, Any],
                              umap_time: float,
                              tsne_time: float) -> pd.DataFrame:
        """
        Crée un tableau comparatif des résultats.
        
        Parameters
        ----------
        umap_metrics : Dict[str, float]
            Métriques UMAP
        tsne_metrics : Dict[str, float]
            Métriques t-SNE
        umap_clustering : Dict[str, Any]
            Résultats clustering UMAP
        tsne_clustering : Dict[str, Any]
            Résultats clustering t-SNE
        umap_time : float
            Temps UMAP
        tsne_time : float
            Temps t-SNE
            
        Returns
        -------
        df : pd.DataFrame
            Tableau de comparaison
        """
        def get_winner(umap_val, tsne_val, lower_is_better=False):
            if lower_is_better:
                return "UMAP" if umap_val < tsne_val else "t-SNE"
            else:
                return "UMAP" if umap_val > tsne_val else "t-SNE"
        
        data = []
        
        # Métriques principales
        metrics_config = [
            ("Trustworthiness", "trustworthiness", False),
            ("Continuity", "continuity", False),
            ("KNN Accuracy", "knn_accuracy", False),
            ("Shepard Correlation", "shepard_correlation", False),
            ("Silhouette Score", "silhouette_score", False),
            ("ARI", "ari_score", False),
            ("NMI", "nmi_score", False)
        ]
        
        for display_name, metric_key, lower_better in metrics_config:
            umap_val = umap_metrics.get(metric_key, 0.0)
            tsne_val = tsne_metrics.get(metric_key, 0.0)
            
            data.append({
                "Métrique": display_name,
                "UMAP": f"{umap_val:.4f}",
                "t-SNE": f"{tsne_val:.4f}",
                "Meilleur": get_winner(umap_val, tsne_val, lower_better)
            })
        
        # Temps d'exécution
        data.append({
            "Métrique": "Temps d'exécution (s)",
            "UMAP": f"{umap_time:.2f}",
            "t-SNE": f"{tsne_time:.2f}",
            "Meilleur": get_winner(umap_time, tsne_time, True)
        })
        
        # Clustering
        umap_n_clusters = umap_clustering.get('n_clusters', 0)
        tsne_n_clusters = tsne_clustering.get('n_clusters', 0)
        
        data.append({
            "Métrique": "Nombre de clusters",
            "UMAP": str(umap_n_clusters),
            "t-SNE": str(tsne_n_clusters),
            "Meilleur": "-"
        })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def format_metric_value(value: float, metric_name: str) -> str:
        """
        Formate une valeur de métrique pour l'affichage.
        
        Parameters
        ----------
        value : float
            Valeur à formater
        metric_name : str
            Nom de la métrique
            
        Returns
        -------
        formatted : str
            Valeur formatée
        """
        if np.isnan(value):
            return "N/A"
        elif np.isinf(value):
            return "∞"
        elif 'time' in metric_name.lower():
            return f"{value:.2f}s"
        elif 'stress' in metric_name.lower():
            return f"{value:.6f}"
        else:
            return f"{value:.4f}"

class ExportManager:
    """Gestionnaire pour l'export/import de sessions."""
    
    @staticmethod
    def export_session(session_state) -> str:
        """
        Exporte l'état de la session vers un CSV.
        
        Parameters
        ----------
        session_state : streamlit.SessionState
            État de la session Streamlit
            
        Returns
        -------
        csv_data : str
            Données CSV prêtes pour téléchargement
        """
        try:
            # Extraction des données principales
            data = {
                'x_umap': session_state.X_umap[:, 0].tolist(),
                'y_umap': session_state.X_umap[:, 1].tolist(),
                'x_tsne': session_state.X_tsne[:, 0].tolist(),
                'y_tsne': session_state.X_tsne[:, 1].tolist(),
                'labels': session_state.y.tolist(),
                'class_names': session_state.class_names
            }
            
            # Métadonnées de session
            metadata = {
                'dataset_choice': session_state.dataset_choice,
                'n_samples': len(session_state.y),
                'export_timestamp': pd.Timestamp.now().isoformat(),
                'umap_time': getattr(session_state, 't_umap', 0),
                'tsne_time': getattr(session_state, 't_tsne', 0)
            }
            
            # Création du DataFrame
            df = pd.DataFrame({
                'x_umap': data['x_umap'],
                'y_umap': data['y_umap'], 
                'x_tsne': data['x_tsne'],
                'y_tsne': data['y_tsne'],
                'labels': data['labels']
            })
            
            # Ajout des métadonnées comme première ligne de commentaire
            csv_buffer = io.StringIO()
            csv_buffer.write(f"# Session Metadata: {json.dumps(metadata)}\n")
            csv_buffer.write(f"# Class Names: {json.dumps(data['class_names'])}\n")
            
            # Export du DataFrame
            df.to_csv(csv_buffer, index=False)
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            st.error(f"Erreur lors de l'export: {str(e)}")
            return ""
    
    @staticmethod
    def import_session(uploaded_file) -> Dict[str, Any]:
        """
        Importe une session depuis un fichier CSV.
        
        Parameters
        ----------
        uploaded_file : streamlit.UploadedFile
            Fichier uploadé
            
        Returns
        -------
        session_data : Dict[str, Any]
            Données de session importées
        """
        try:
            # Lecture du contenu
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.split('\n')
            
            # Extraction des métadonnées depuis les commentaires
            metadata = {}
            class_names = []
            data_start_line = 0
            
            for i, line in enumerate(lines):
                if line.startswith('# Session Metadata:'):
                    metadata = json.loads(line.replace('# Session Metadata: ', ''))
                elif line.startswith('# Class Names:'):
                    class_names = json.loads(line.replace('# Class Names: ', ''))
                elif not line.startswith('#'):
                    data_start_line = i
                    break
            
            # Lecture des données principales
            data_lines = '\n'.join(lines[data_start_line:])
            df = pd.read_csv(io.StringIO(data_lines))
            
            # Construction du dictionnaire de session
            session_data = {
                'X_umap': df[['x_umap', 'y_umap']].values,
                'X_tsne': df[['x_tsne', 'y_tsne']].values,
                'y': df['labels'].values,
                'class_names': class_names,
                'metadata': metadata
            }
            
            return session_data
            
        except Exception as e:
            raise ValueError(f"Erreur lors de l'import: {str(e)}")
    
    @staticmethod
    def export_metrics_report(umap_metrics: Dict[str, float],
                            tsne_metrics: Dict[str, float],
                            comparison_df: pd.DataFrame,
                            filename: str = "metrics_report.html") -> str:
        """
        Exporte un rapport HTML des métriques.
        
        Parameters
        ----------
        umap_metrics : Dict[str, float]
            Métriques UMAP
        tsne_metrics : Dict[str, float]
            Métriques t-SNE
        comparison_df : pd.DataFrame
            Tableau de comparaison
        filename : str, default="metrics_report.html"
            Nom du fichier de sortie
            
        Returns
        -------
        filepath : str
            Chemin du fichier créé
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UMAP vs t-SNE - Rapport de Métriques</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2E86AB; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .winner {{ background-color: #e8f5e8; font-weight: bold; }}
                .timestamp {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>UMAP vs t-SNE - Rapport de Métriques</h1>
            <p class="timestamp">Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Tableau Comparatif</h2>
            {comparison_df.to_html(escape=False, index=False)}
            
            <h2>Métriques Détaillées</h2>
            
            <h3>UMAP</h3>
            <ul>
                {"".join([f"<li><strong>{k}:</strong> {UIUtils.format_metric_value(v, k)}</li>" for k, v in umap_metrics.items()])}
            </ul>
            
            <h3>t-SNE</h3>
            <ul>
                {"".join([f"<li><strong>{k}:</strong> {UIUtils.format_metric_value(v, k)}</li>" for k, v in tsne_metrics.items()])}
            </ul>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename

class ColorManager:
    """Gestionnaire de palettes de couleurs cohérentes."""
    
    # Palette principale (optimisée pour distinguabilité)
    PRIMARY_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Palette pour daltoniens
    COLORBLIND_PALETTE = [
        '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#7f7f7f'
    ]
    
    @classmethod
    def get_palette(cls, n_colors: int, colorblind_friendly: bool = False) -> List[str]:
        """
        Retourne une palette de couleurs adaptée.
        
        Parameters
        ----------
        n_colors : int
            Nombre de couleurs nécessaires
        colorblind_friendly : bool, default=False
            Utiliser une palette adaptée aux daltoniens
            
        Returns
        -------
        palette : List[str]
            Liste de couleurs hexadécimales
        """
        base_palette = cls.COLORBLIND_PALETTE if colorblind_friendly else cls.PRIMARY_PALETTE
        
        if n_colors <= len(base_palette):
            return base_palette[:n_colors]
        else:
            # Extension de la palette si nécessaire
            extended_palette = base_palette.copy()
            
            # Génération de couleurs additionnelles par variation
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            try:
                additional_colors = plt.cm.Set3(np.linspace(0, 1, n_colors - len(base_palette)))
                extended_palette.extend([mcolors.rgb2hex(color[:3]) for color in additional_colors])
            except:
                # Fallback si erreur matplotlib
                for i in range(n_colors - len(base_palette)):
                    extended_palette.append('#888888')  # Gris par défaut
            
            return extended_palette[:n_colors]
    
    @staticmethod
    def create_gradient(start_color: str, end_color: str, n_steps: int) -> List[str]:
        """
        Crée un dégradé entre deux couleurs.
        
        Parameters
        ----------
        start_color : str
            Couleur de début (hex)
        end_color : str
            Couleur de fin (hex)
        n_steps : int
            Nombre d'étapes du dégradé
            
        Returns
        -------
        gradient : List[str]
            Liste de couleurs du dégradé
        """
        try:
            import matplotlib.colors as mcolors
            
            # Conversion en RGB
            start_rgb = mcolors.hex2color(start_color)
            end_rgb = mcolors.hex2color(end_color)
            
            # Génération du dégradé
            gradient = []
            for i in range(n_steps):
                t = i / (n_steps - 1) if n_steps > 1 else 0
                
                r = start_rgb[0] + t * (end_rgb[0] - start_rgb[0])
                g = start_rgb[1] + t * (end_rgb[1] - start_rgb[1])
                b = start_rgb[2] + t * (end_rgb[2] - start_rgb[2])
                
                gradient.append(mcolors.rgb2hex((r, g, b)))
            
            return gradient
        
        except:
            # Fallback simple
            return [start_color] * n_steps


def save_best_params(dataset: str, model: str, params: dict, metric: str, score: float) -> Path:
    """
    Sauvegarde les meilleurs hyperparamètres pour un modèle et dataset donné.

    Parameters
    ----------
    dataset : str
        Nom du dataset
    model : str
        Nom du modèle (UMAP, t-SNE, etc.)
    params : dict
        Dictionnaire des hyperparamètres
    metric : str
        Métrique utilisée pour l'évaluation
    score : float
        Score obtenu

    Returns
    -------
    filepath : Path
        Chemin du fichier sauvegardé
    """
    # Créer le dossier si nécessaire
    output_dir = Path("outputs/runs") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Préparer les données à sauvegarder
    data = {
        "model": model,
        "dataset": dataset,
        "params": params,
        "metric": metric,
        "score": score,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # Sauvegarder en JSON
    filepath = output_dir / f"{model}_best.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


def load_best_params(dataset: str, model: str) -> Optional[dict]:
    """
    Charge les meilleurs hyperparamètres sauvegardés pour un modèle et dataset donné.

    Parameters
    ----------
    dataset : str
        Nom du dataset
    model : str
        Nom du modèle (UMAP, t-SNE, etc.)

    Returns
    -------
    params : dict or None
        Dictionnaire des hyperparamètres si trouvé, sinon None
    """
    filepath = Path("outputs/runs") / dataset / f"{model}_best.json"

    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data.get("params", None)
        except Exception as e:
            warnings.warn(f"Erreur lors du chargement des paramètres: {e}")
            return None

    return None