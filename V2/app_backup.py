# app.py - Version avec modes Expérimenter/Comparer
import time
from typing import Any, Dict, List, Tuple, Optional
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from mpl_toolkits.mplot3d import Axes3D

from core.clustering import ClusteringManager
from core.data import DataManager
from core.embeddings import EmbeddingManager
from core.metrics import MetricsCalculator
from core.plots import PlotManager
from core.sweep import SweepManager
from core.utils import UIUtils, seed_everything, save_best_params, load_best_params
from core.vectorizers import VectorizerManager

st.set_page_config(
    page_title="Dimensionality Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialise le state de la session."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'embeddings_computed' not in st.session_state:
        st.session_state.embeddings_computed = False
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'sweep_results' not in st.session_state:
        st.session_state.sweep_results = None


def render_header() -> str:
    """Affiche l'en-tête et retourne le mode sélectionné."""
    st.title("🔍 Dimensionality Lab")
    st.markdown("*Laboratoire interactif de réduction de dimension*")
    return None  # Éviter tout retour de DeltaGenerator


def render_sidebar_mode() -> str:
    """Sélection du mode principal."""
    st.sidebar.header("🎯 Mode d'Exécution")
    mode = st.sidebar.radio(
        "Choisir le mode",
        ["🔬 Expérimenter un modèle", "⚖️ Comparer des modèles"],
        index=0,
        help="Expérimenter: tester un modèle avec sweep | Comparer: comparer plusieurs modèles"
    )
    return mode


def render_data_section(data_manager: DataManager) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Section de sélection et chargement des données."""
    st.sidebar.header("📊 Configuration des Données")

    # Liste des datasets
    builtin_datasets = ["Digits (MNIST)", "20 Newsgroups"]
    custom_datasets = data_manager.list_datasets("data")
    all_datasets = builtin_datasets + custom_datasets

    dataset_choice = st.sidebar.selectbox(
        "Dataset",
        options=all_datasets,
        index=0,
        help="Sélectionnez le dataset à analyser"
    )

    # Sous-échantillonnage
    subsample_size = st.sidebar.slider(
        "Taille d'échantillon",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
        help="Nombre d'échantillons à utiliser"
    )

    # Options spécifiques texte
    svd_components = None
    if dataset_choice == "20 Newsgroups" or dataset_choice in custom_datasets:
        with st.sidebar.expander("Options avancées"):
            svd_components = st.sidebar.slider(
                "Composantes SVD (texte)",
                min_value=50,
                max_value=500,
                value=300,
                step=50,
                help="Pour les données textuelles uniquement"
            )

    # Bouton Inspecter
    inspect_data = st.sidebar.button("🔍 Inspecter le dataset")

    # Chargement des données
    if dataset_choice != st.session_state.current_dataset:
        st.session_state.data_loaded = False
        st.session_state.embeddings_computed = False
        st.session_state.current_dataset = dataset_choice

    if not st.session_state.data_loaded:
        with st.spinner(f"Chargement du dataset {dataset_choice}..."):
            X, y, class_names, meta = data_manager.load_dataset(dataset_choice)

            # Sous-échantillonnage reproductible
            if len(X) > subsample_size:
                np.random.seed(42)
                indices = np.random.choice(len(X), size=subsample_size, replace=False)
                X = X[indices]
                y = y[indices]

            # Mise à jour des métadonnées
            meta['n_samples'] = X.shape[0]

            # Stockage en session
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.class_names = class_names
            st.session_state.meta = meta
            st.session_state.dataset_choice = dataset_choice
            st.session_state.data_loaded = True

    X = st.session_state.X
    y = st.session_state.y
    class_names = st.session_state.class_names
    meta = st.session_state.meta

    # Affichage des infos
    st.sidebar.success(DataManager.describe_dataset(meta))

    # Inspection détaillée
    if inspect_data:
        with st.expander("📊 Détails du dataset", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Échantillons", meta['n_samples'])
                st.metric("Type", meta['type'])
            with col2:
                st.metric("Features", meta['n_features'])
                st.metric("Classes", meta['n_classes'])
            with col3:
                if 'timestamp' in meta:
                    st.metric("Chargé", meta.get('timestamp', 'N/A'))

            # Distribution des classes
            if len(np.unique(y)) < 50:
                fig, ax = plt.subplots(figsize=(8, 3))
                unique, counts = np.unique(y, return_counts=True)
                ax.bar(unique, counts, color='steelblue')
                ax.set_xlabel("Classe")
                ax.set_ylabel("Nombre d'échantillons")
                ax.set_title("Distribution des classes")
                st.pyplot(fig)
                plt.close(fig)

    return X, y, class_names, meta


def render_experiment_mode(X, y, class_names, meta, data_manager):
    """Mode Expérimenter un modèle."""

    # Section Vectorisation
    st.sidebar.header("🧩 Vectorisation")
    vectorizer_manager = VectorizerManager()

    # Obtenir les méthodes disponibles selon le type de données
    available_methods = vectorizer_manager.get_methods_for_type(meta['type'])

    vectorization_method = st.sidebar.selectbox(
        "Méthode de vectorisation",
        options=available_methods,
        index=0,
        help="Choisissez la méthode de prétraitement des données"
    )

    # Options spécifiques pour certaines méthodes
    svd_components = 300  # Valeur par défaut
    if "SVD" in vectorization_method or "PCA" in vectorization_method:
        with st.sidebar.expander("Options de vectorisation"):
            svd_components = st.slider(
                "Nombre de composantes",
                min_value=50,
                max_value=500,
                value=300,
                step=50,
                help="Nombre de composantes pour la réduction dimensionnelle"
            )

    st.sidebar.header("⚙️ Configuration du Modèle")

    # Sélection du modèle
    model_choice = st.sidebar.selectbox(
        "Modèle",
        options=["UMAP", "t-SNE", "PCA", "Isomap", "Spectral"],
        index=0
    )

    params = {}

    # Paramètres UMAP
    if model_choice == "UMAP":
        st.sidebar.subheader("🎯 Paramètres UMAP")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            params['n_neighbors'] = st.slider("n_neighbors", 2, 200, 15)
            params['min_dist'] = st.slider("min_dist", 0.0, 1.0, 0.1, 0.01)
        with col2:
            params['spread'] = st.slider("spread", 0.1, 3.0, 1.0, 0.1)
            params['n_epochs'] = st.slider("n_epochs", 50, 1000, 200, 50)

        # Paramètres avancés UMAP
        with st.sidebar.expander("🔧 Paramètres avancés UMAP"):
            # Métrique de distance
            default_metric = 'cosine' if meta['type'] == 'text' else 'euclidean'
            params['metric'] = st.selectbox(
                "metric (distance)",
                options=["euclidean", "manhattan", "chebyshev", "minkowski",
                        "canberra", "braycurtis", "cosine", "correlation"],
                index=["euclidean", "manhattan", "chebyshev", "minkowski",
                      "canberra", "braycurtis", "cosine", "correlation"].index(default_metric),
                help="cosine souvent meilleur sur texte, euclidean sur images"
            )

            # Méthode d'initialisation
            params['init'] = st.selectbox(
                "init (initialisation)",
                options=["spectral", "random", "pca"],
                index=0,
                help="spectral par défaut, pca utile si PCA préalable"
            )

            # Learning rate
            params['learning_rate'] = st.slider(
                "learning_rate",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="plus grand = convergence rapide mais bruyante"
            )

            # Nombre de composantes
            params['n_components'] = st.selectbox(
                "n_components",
                options=[2, 3],
                index=0,
                help="2D ou 3D embedding"
            )

    # Paramètres t-SNE
    elif model_choice == "t-SNE":
        st.sidebar.subheader("📊 Paramètres t-SNE")

        col3, col4 = st.sidebar.columns(2)
        with col3:
            params['perplexity'] = st.slider("perplexity", 5, 100, 30)
            params['max_iter'] = st.slider("max_iter", 250, 2000, 1000, 50)
        with col4:
            params['learning_rate'] = st.selectbox("learning_rate", ["auto", 10, 50, 200, 1000], 0)
            params['pca_prenorm'] = st.checkbox("PCA pré-normalisation", True)

    # Autres modèles
    elif model_choice == "PCA":
        params['n_components'] = 2

    elif model_choice == "Isomap":
        params['n_neighbors'] = st.sidebar.slider("n_neighbors", 2, 50, 10)
        params['n_components'] = 2

    elif model_choice == "Spectral":
        params['n_neighbors'] = st.sidebar.slider("n_neighbors", 2, 50, 10)
        params['affinity'] = st.sidebar.selectbox("affinity", ["nearest_neighbors", "rbf"], 0)
        params['n_components'] = 2

    # Section Sweep
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔥 Sweep Hyperparamètres"):
        sweep_metric = st.selectbox(
            "Métrique d'optimisation",
            ["trustworthiness", "knn_accuracy"],
            index=0
        )

        if model_choice == "UMAP":
            st.markdown("**Grille de recherche UMAP**")
            n_neighbors_list = st.text_input(
                "n_neighbors (liste)",
                "5, 15, 30, 50",
                help="Valeurs séparées par virgule"
            )
            min_dist_list = st.text_input(
                "min_dist (liste)",
                "0.0, 0.1, 0.3, 0.5",
                help="Valeurs séparées par virgule"
            )

            # Options avancées pour le sweep
            include_advanced = st.checkbox("Inclure paramètres avancés dans le sweep")
            if include_advanced:
                metric_list = st.multiselect(
                    "Métriques à tester",
                    ["euclidean", "manhattan", "cosine", "correlation"],
                    default=["euclidean", "cosine"] if meta['type'] == 'text' else ["euclidean"]
                )
                init_list = st.multiselect(
                    "Initialisations à tester",
                    ["spectral", "random", "pca"],
                    default=["spectral"]
                )

        elif model_choice == "t-SNE":
            st.markdown("**Grille de recherche t-SNE**")
            perplexity_list = st.text_input(
                "perplexity (liste)",
                "5, 15, 30, 50",
                help="Valeurs séparées par virgule"
            )
            learning_rate_list = st.text_input(
                "learning_rate (liste)",
                "auto, 200, 500",
                help="Valeurs séparées par virgule"
            )

        launch_sweep = st.button("🚀 Lancer Sweep", type="secondary")

    # Boutons d'action principaux
    st.sidebar.markdown("---")
    compute_embedding = st.sidebar.button("💫 Calculer Embedding", type="primary")

    # Zone principale
    if compute_embedding or launch_sweep:
        embedding_manager = EmbeddingManager()

        # Préparation des données avec VectorizerManager
        with st.spinner(f"Vectorisation des données ({vectorization_method})..."):
            X_processed = vectorizer_manager.vectorize(X, meta['type'], vectorization_method, svd_components)
            st.session_state.X_processed = X_processed
            vectorization_metadata = vectorizer_manager.get_metadata()

        # Calcul de l'embedding simple
        if compute_embedding:
            with st.spinner(f"Calcul {model_choice}..."):
                t_start = time.time()

                if model_choice == "UMAP":
                    X_embedded, _ = embedding_manager.compute_umap(X_processed, params)
                elif model_choice == "t-SNE":
                    X_embedded, _ = embedding_manager.compute_tsne(X_processed, params)
                elif model_choice == "PCA":
                    pca = PCA(n_components=2, random_state=42)
                    X_embedded = pca.fit_transform(X_processed)
                elif model_choice == "Isomap":
                    iso = Isomap(n_neighbors=params['n_neighbors'], n_components=2)
                    X_embedded = iso.fit_transform(X_processed)
                elif model_choice == "Spectral":
                    spec = SpectralEmbedding(
                        n_components=2,
                        n_neighbors=params['n_neighbors'],
                        affinity=params['affinity'],
                        random_state=42
                    )
                    X_embedded = spec.fit_transform(X_processed)

                t_elapsed = time.time() - t_start

            # Métriques
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.calculate_all_metrics(X, X_embedded, y)

            # Clustering
            clustering_mgr = ClusteringManager()
            clusters = clustering_mgr.cluster_embeddings(X_embedded, y, len(class_names))

            # Affichage
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"🎯 {model_choice} Embedding")
                plot_mgr = PlotManager(use_plotly=False)
                fig = plot_mgr.plot_embedding(X_embedded, y, class_names, f"{model_choice} (t={t_elapsed:.2f}s)")
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.subheader("📊 Métriques")
                UIUtils().display_metrics_card(model_choice, metrics, clusters)

            # Affichage des détails de vectorisation
            with st.expander("📋 Détails de Vectorisation", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Méthode", vectorization_metadata.get('method', 'N/A'))
                    st.metric("Forme d'entrée", str(vectorization_metadata.get('input_shape', 'N/A')))
                with col2:
                    st.metric("Forme de sortie", str(vectorization_metadata.get('output_shape', 'N/A')))
                    if 'sparsity' in vectorization_metadata:
                        st.metric("Sparsité", f"{vectorization_metadata['sparsity']:.1%}")
                with col3:
                    if 'explained_variance_ratio' in vectorization_metadata:
                        st.metric("Variance expliquée", f"{vectorization_metadata['explained_variance_ratio']:.1%}")
                    if 'hf_model' in vectorization_metadata:
                        st.metric("Modèle HF", vectorization_metadata['hf_model'])
                    if 'svd_components' in vectorization_metadata:
                        st.metric("Composantes SVD", vectorization_metadata['svd_components'])
                    if 'pca_components' in vectorization_metadata:
                        st.metric("Composantes PCA", vectorization_metadata['pca_components'])

            st.session_state.last_embedding = X_embedded
            st.session_state.last_metrics = metrics
            st.session_state.last_model = model_choice
            st.session_state.vectorization_metadata = vectorization_metadata

            # Add advanced results panel for single embedding
            embeddings_dict = {model_choice: X_embedded}
            render_advanced_results_panel(X, X_processed, embeddings_dict, y, class_names, meta)

        # Sweep
        if launch_sweep and model_choice in ["UMAP", "t-SNE"]:
            sweep_mgr = SweepManager()

            # Parse des grilles
            if model_choice == "UMAP":
                param_grid = {
                    'n_neighbors': [int(x.strip()) for x in n_neighbors_list.split(',')],
                    'min_dist': [float(x.strip()) for x in min_dist_list.split(',')]
                }

                # Ajouter les paramètres avancés si demandé
                if 'include_advanced' in locals() and include_advanced:
                    if 'metric_list' in locals() and metric_list:
                        param_grid['metric'] = metric_list
                    if 'init_list' in locals() and init_list:
                        param_grid['init'] = init_list
            else:
                lr_values = []
                for x in learning_rate_list.split(','):
                    val = x.strip()
                    lr_values.append('auto' if val == 'auto' else int(val))

                param_grid = {
                    'perplexity': [int(x.strip()) for x in perplexity_list.split(',')],
                    'learning_rate': lr_values
                }

            with st.spinner(f"Sweep {model_choice} en cours..."):
                if model_choice == "UMAP":
                    results, heatmap_fig, best_params = sweep_mgr.umap_parameter_sweep(
                        X_processed, y, sweep_metric, param_grid
                    )
                else:
                    results, heatmap_fig, best_params = sweep_mgr.tsne_parameter_sweep(
                        X_processed, y, sweep_metric, param_grid
                    )

            # Affichage des résultats
            st.subheader("🔥 Résultats du Parameter Sweep")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(heatmap_fig)
                plt.close(heatmap_fig)

            with col2:
                if best_params:
                    st.success(f"**Meilleur {sweep_metric}:**")
                    best_score = max(results[sweep_metric])
                    st.metric(sweep_metric, f"{best_score:.4f}")

                    st.info("**Meilleurs paramètres:**")
                    for k, v in best_params.items():
                        if k not in ['metric', 'n_components', 'random_state', 'init', 'pca_prenorm']:
                            st.write(f"• {k}: {v}")

                    # Boutons d'action
                    if st.button("✅ Appliquer ces paramètres"):
                        st.session_state.best_params = best_params
                        st.rerun()

                    if st.button("💾 Sauvegarder comme meilleurs"):
                        filepath = save_best_params(
                            meta['name'],
                            model_choice,
                            best_params,
                            sweep_metric,
                            best_score
                        )
                        st.success(f"Sauvegardé: {filepath}")

            st.session_state.sweep_results = (results, best_params)

            # Add advanced results panel for sweep results with best embedding
            if best_params:
                # Compute best embedding for advanced analysis
                with st.spinner(f"Calcul de l'embedding optimal pour l'analyse avancée..."):
                    if model_choice == "UMAP":
                        best_X_embedded, _ = embedding_manager.compute_umap(X_processed, best_params)
                    else:  # t-SNE
                        best_X_embedded, _ = embedding_manager.compute_tsne(X_processed, best_params)

                embeddings_dict = {f"{model_choice} (Optimal)": best_X_embedded}
                render_advanced_results_panel(X, X_processed, embeddings_dict, y, class_names, meta)

    return None  # Éviter tout retour de DeltaGenerator


def render_compare_mode(X, y, class_names, meta, data_manager):
    """Mode Comparer des modèles."""

    # Section Vectorisation
    st.sidebar.header("🧩 Vectorisation")
    vectorizer_manager = VectorizerManager()

    # Obtenir les méthodes disponibles selon le type de données
    available_methods = vectorizer_manager.get_methods_for_type(meta['type'])

    vectorization_method = st.sidebar.selectbox(
        "Méthode de vectorisation",
        options=available_methods,
        index=0,
        help="Choisissez la méthode de prétraitement des données"
    )

    # Options spécifiques pour certaines méthodes
    svd_components = 300  # Valeur par défaut
    if "SVD" in vectorization_method or "PCA" in vectorization_method:
        with st.sidebar.expander("Options de vectorisation"):
            svd_components = st.slider(
                "Nombre de composantes",
                min_value=50,
                max_value=500,
                value=300,
                step=50,
                help="Nombre de composantes pour la réduction dimensionnelle"
            )

    st.sidebar.header("⚙️ Configuration de la Comparaison")

    # Sélection des modèles
    models_to_compare = st.sidebar.multiselect(
        "Modèles à comparer",
        ["PCA", "UMAP", "t-SNE", "Isomap", "Spectral"],
        default=["PCA", "UMAP", "t-SNE"]
    )

    # Option pour utiliser les meilleurs params
    use_best = st.sidebar.checkbox(
        "📁 Utiliser les meilleurs hyperparamètres sauvegardés",
        value=False,
        help="Si disponibles, charge les paramètres optimisés pour ce dataset"
    )

    # Paramètres manuels si pas use_best
    params_dict = {}
    if not use_best:
        with st.sidebar.expander("Paramètres des modèles"):
            if "UMAP" in models_to_compare:
                st.markdown("**UMAP**")
                default_metric = 'cosine' if meta['type'] == 'text' else 'euclidean'
                params_dict["UMAP"] = {
                    'n_neighbors': st.slider("UMAP n_neighbors", 2, 100, 15),
                    'min_dist': st.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.01),
                    'metric': st.selectbox(
                        "UMAP metric",
                        ["euclidean", "manhattan", "cosine", "correlation"],
                        index=["euclidean", "manhattan", "cosine", "correlation"].index(default_metric)
                    ),
                    'init': st.selectbox("UMAP init", ["spectral", "random", "pca"], 0),
                    'learning_rate': st.slider("UMAP learning_rate", 0.01, 10.0, 1.0, 0.1),
                    'n_components': 2
                }

            if "t-SNE" in models_to_compare:
                st.markdown("**t-SNE**")
                params_dict["t-SNE"] = {
                    'perplexity': st.slider("t-SNE perplexity", 5, 100, 30),
                    'learning_rate': st.selectbox("t-SNE learning_rate", ["auto", 200, 500], 0),
                    'max_iter': 1000,
                    'pca_prenorm': True
                }

            if "Isomap" in models_to_compare:
                st.markdown("**Isomap**")
                params_dict["Isomap"] = {
                    'n_neighbors': st.slider("Isomap n_neighbors", 2, 50, 10)
                }

            if "Spectral" in models_to_compare:
                st.markdown("**Spectral**")
                params_dict["Spectral"] = {
                    'n_neighbors': st.slider("Spectral n_neighbors", 2, 50, 10),
                    'affinity': st.selectbox("Spectral affinity", ["nearest_neighbors", "rbf"], 0)
                }

    # Bouton de comparaison
    st.sidebar.markdown("---")
    run_comparison = st.sidebar.button("🏁 Lancer la Comparaison", type="primary")

    if run_comparison:
        embedding_manager = EmbeddingManager()
        metrics_calc = MetricsCalculator()

        # Préparation des données avec VectorizerManager
        with st.spinner(f"Vectorisation des données ({vectorization_method})..."):
            X_processed = vectorizer_manager.vectorize(X, meta['type'], vectorization_method, svd_components)
            st.session_state.X_processed = X_processed
            vectorization_metadata = vectorizer_manager.get_metadata()

        results = []
        embeddings = {}

        # Calcul pour chaque modèle
        progress_bar = st.progress(0)
        for i, model in enumerate(models_to_compare):
            with st.spinner(f"Calcul {model}..."):
                # Récupération des paramètres
                if use_best:
                    loaded_params = load_best_params(meta['name'], model)
                    if loaded_params:
                        params = loaded_params
                        st.info(f"✅ Paramètres optimisés chargés pour {model}")
                    else:
                        params = params_dict.get(model, {})
                        st.warning(f"⚠️ Pas de paramètres sauvegardés pour {model}, utilisation des valeurs par défaut")
                else:
                    params = params_dict.get(model, {})

                # Calcul de l'embedding
                t_start = time.time()

                if model == "UMAP":
                    X_embedded, _ = embedding_manager.compute_umap(X_processed, params)
                elif model == "t-SNE":
                    X_embedded, _ = embedding_manager.compute_tsne(X_processed, params)
                elif model == "PCA":
                    pca = PCA(n_components=2, random_state=42)
                    X_embedded = pca.fit_transform(X_processed)
                elif model == "Isomap":
                    n_neighbors = params.get('n_neighbors', 10)
                    iso = Isomap(n_neighbors=n_neighbors, n_components=2)
                    X_embedded = iso.fit_transform(X_processed)
                elif model == "Spectral":
                    n_neighbors = params.get('n_neighbors', 10)
                    affinity = params.get('affinity', 'nearest_neighbors')
                    spec = SpectralEmbedding(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        affinity=affinity,
                        random_state=42
                    )
                    X_embedded = spec.fit_transform(X_processed)

                t_elapsed = time.time() - t_start

                # Calcul des métriques
                metrics = metrics_calc.calculate_all_metrics(X, X_embedded, y)

                results.append({
                    'Modèle': model,
                    'Temps (s)': t_elapsed,
                    'Trustworthiness': metrics['trustworthiness'],
                    'KNN Accuracy': metrics['knn_accuracy'],
                    'Shepard r': metrics.get('shepard_correlation', np.nan),
                    'Continuity': metrics.get('continuity', np.nan),
                    'Params optimisés': '✅' if use_best and loaded_params else '❌'
                })

                embeddings[model] = X_embedded

            progress_bar.progress((i + 1) / len(models_to_compare))

        progress_bar.empty()

        # Affichage des détails de vectorisation
        with st.expander("📋 Détails de Vectorisation", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Méthode", vectorization_metadata.get('method', 'N/A'))
                st.metric("Forme d'entrée", str(vectorization_metadata.get('input_shape', 'N/A')))
            with col2:
                st.metric("Forme de sortie", str(vectorization_metadata.get('output_shape', 'N/A')))
                if 'sparsity' in vectorization_metadata:
                    st.metric("Sparsité", f"{vectorization_metadata['sparsity']:.1%}")
            with col3:
                if 'explained_variance_ratio' in vectorization_metadata:
                    st.metric("Variance expliquée", f"{vectorization_metadata['explained_variance_ratio']:.1%}")
                if 'hf_model' in vectorization_metadata:
                    st.metric("Modèle HF", vectorization_metadata['hf_model'])
                if 'svd_components' in vectorization_metadata:
                    st.metric("Composantes SVD", vectorization_metadata['svd_components'])
                if 'pca_components' in vectorization_metadata:
                    st.metric("Composantes PCA", vectorization_metadata['pca_components'])

        # Tableau comparatif
        st.subheader("📊 Tableau Comparatif")
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Trustworthiness', ascending=False)

        # Style du tableau
        def highlight_best(s):
            is_max = s == s.max()
            is_min = s == s.min()
            styles = []
            for i, (mx, mn) in enumerate(zip(is_max, is_min)):
                if s.name == 'Temps (s)' and mn:
                    styles.append('background-color: lightgreen')
                elif mx and s.name != 'Temps (s)':
                    styles.append('background-color: lightgreen')
                else:
                    styles.append('')
            return styles

        numeric_cols = ['Temps (s)', 'Trustworthiness', 'KNN Accuracy', 'Shepard r', 'Continuity']
        styled_df = df_results.style.apply(highlight_best, subset=numeric_cols)
        st.dataframe(styled_df, use_container_width=True)

        # Graphiques côte à côte
        st.subheader("🎨 Visualisations")
        plot_mgr = PlotManager(use_plotly=False)

        n_models = len(models_to_compare)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                model_idx = row * n_cols + col_idx
                if model_idx < n_models:
                    model = models_to_compare[model_idx]
                    with cols[col_idx]:
                        fig = plot_mgr.plot_embedding(
                            embeddings[model],
                            y,
                            class_names,
                            f"{model}"
                        )
                        st.pyplot(fig)
                        plt.close(fig)

        # Graphiques de métriques
        st.subheader("📈 Comparaison des Métriques")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_results['Modèle'], df_results['Temps (s)'], color='steelblue')
            ax.set_ylabel("Temps (s)")
            ax.set_title("Temps d'exécution")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_results['Modèle'], df_results['Trustworthiness'], color='coral')
            ax.set_ylabel("Trustworthiness")
            ax.set_title("Préservation locale")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_results['Modèle'], df_results['KNN Accuracy'], color='lightgreen')
            ax.set_ylabel("KNN Accuracy")
            ax.set_title("Précision KNN")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

        # Add advanced results panel for all computed embeddings
        render_advanced_results_panel(X, X_processed, embeddings, y, class_names, meta)

    return None  # Éviter tout retour de DeltaGenerator


def create_3d_visualization(X_processed, y, class_names, method='PCA'):
    """Create a 3D visualization using PCA, UMAP, or t-SNE."""
    if method == 'PCA':
        reducer = PCA(n_components=3, random_state=42)
        embedding_3d = reducer.fit_transform(X_processed)
        title = f"3D PCA"
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_3d = reducer.fit_transform(X_processed)
        title = f"3D UMAP"
    elif method == 't-SNE':
        reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        embedding_3d = reducer.fit_transform(X_processed)
        title = f"3D t-SNE"

    # Create 3D scatter plot with plotly
    fig = go.Figure()

    # Color palette
    colors = px.colors.qualitative.Set1[:len(class_names)]

    for i, class_name in enumerate(class_names):
        mask = y == i
        fig.add_trace(go.Scatter3d(
            x=embedding_3d[mask, 0],
            y=embedding_3d[mask, 1],
            z=embedding_3d[mask, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            name=str(class_name),
            text=[f'Class: {class_name}'] * np.sum(mask),
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        width=700,
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
    )

    return fig


def plot_trustworthiness_vs_k(X, X_embedded_dict, y, max_k=50):
    """Plot trustworthiness vs k for different embeddings."""
    from core.metrics import MetricsCalculator
    metrics_calc = MetricsCalculator()

    k_values = range(1, min(max_k + 1, len(X) // 4))

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, X_embedded in X_embedded_dict.items():
        trustworthiness_scores = []

        for k in k_values:
            trust_score = metrics_calc.trustworthiness(X, X_embedded, k=k)
            trustworthiness_scores.append(trust_score)

        ax.plot(k_values, trustworthiness_scores, marker='o', linewidth=2,
                markersize=4, label=method_name)

    ax.set_xlabel('k (number of nearest neighbors)')
    ax.set_ylabel('Trustworthiness')
    ax.set_title('Trustworthiness vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    return fig


def create_knn_confusion_matrix(X_embedded, y, class_names, k=5):
    """Create KNN confusion matrix on embeddings."""
    # Train KNN classifier on embeddings
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_embedded, y)
    y_pred = knn.predict(X_embedded)

    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'KNN Confusion Matrix (k={k}) on Embeddings')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    return fig, accuracy


def export_embeddings_csv(embeddings_dict, y, class_names):
    """Export embeddings to CSV format."""
    # Create a comprehensive dataframe
    data_list = []

    for method_name, embedding in embeddings_dict.items():
        for i, (point, label) in enumerate(zip(embedding, y)):
            row = {
                'Sample_ID': i,
                'Method': method_name,
                'Class_Label': label,
                'Class_Name': class_names[label] if label < len(class_names) else f'Class_{label}',
                'Dim1': point[0],
                'Dim2': point[1]
            }
            if embedding.shape[1] > 2:
                row['Dim3'] = point[2]
            data_list.append(row)

    df = pd.DataFrame(data_list)

    # Convert to CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_content = buffer.getvalue()
    buffer.close()

    return csv_content


def create_digits_3d_relief(X, y, class_names, selected_class=0):
    """Create 3D relief visualization for digits dataset."""
    if 'Digits' not in str(class_names[0]) and len(X[0]) != 64:
        return None, "Cette visualisation n'est disponible que pour le dataset Digits (8x8 pixels)"

    # Get samples for the selected class
    class_indices = np.where(y == selected_class)[0]
    if len(class_indices) == 0:
        return None, f"Aucun échantillon trouvé pour la classe {selected_class}"

    # Take first sample of the selected class
    sample = X[class_indices[0]].reshape(8, 8)

    # Create coordinate grids
    x = np.arange(8)
    y = np.arange(8)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X_grid,
        y=Y_grid,
        z=sample,
        colorscale='Viridis',
        showscale=True
    )])

    fig.update_layout(
        title=f'Relief 3D - Chiffre {selected_class}',
        scene=dict(
            xaxis_title='Pixel X',
            yaxis_title='Pixel Y',
            zaxis_title='Intensité',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=600,
        height=500
    )

    return fig, None


def render_advanced_results_panel(X, X_processed, embeddings_dict, y, class_names, meta):
    """Render the advanced results panel with all features."""
    st.markdown("---")

    with st.expander("🔎 Résultats avancés", expanded=False):
        st.markdown("### Analyses approfondies des embeddings")

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visualizations 3D",
            "📈 Trustworthiness vs k",
            "🎯 Matrices de Confusion KNN",
            "💾 Export CSV",
            "🏔️ Relief 3D (Digits)"
        ])

        # Tab 1: 3D Visualizations
        with tab1:
            st.markdown("#### Visualisations 3D des embeddings")

            col1, col2 = st.columns([1, 2])
            with col1:
                viz_method = st.selectbox(
                    "Méthode 3D:",
                    ["PCA", "UMAP", "t-SNE"],
                    key="3d_viz_method"
                )

                if st.button("Générer visualisation 3D", key="generate_3d"):
                    with st.spinner(f"Génération de la visualisation 3D {viz_method}..."):
                        fig_3d = create_3d_visualization(X_processed, y, class_names, viz_method)
                        st.session_state.fig_3d = fig_3d

            with col2:
                if 'fig_3d' in st.session_state:
                    st.plotly_chart(st.session_state.fig_3d, use_container_width=True)

        # Tab 2: Trustworthiness vs k curves
        with tab2:
            st.markdown("#### Courbes Trustworthiness vs k")

            col1, col2 = st.columns([1, 2])
            with col1:
                max_k = st.slider("Valeur k maximale", 5, 100, 50, key="max_k_trust")

                if st.button("Calculer courbes", key="calc_trust_curves") and embeddings_dict:
                    with st.spinner("Calcul des courbes de trustworthiness..."):
                        fig_trust = plot_trustworthiness_vs_k(X, embeddings_dict, y, max_k)
                        st.session_state.fig_trust = fig_trust

            with col2:
                if 'fig_trust' in st.session_state:
                    st.pyplot(st.session_state.fig_trust)

        # Tab 3: KNN Confusion matrices
        with tab3:
            st.markdown("#### Matrices de Confusion KNN sur les Embeddings")

            if embeddings_dict:
                col1, col2 = st.columns([1, 2])
                with col1:
                    selected_method = st.selectbox(
                        "Méthode d'embedding:",
                        list(embeddings_dict.keys()),
                        key="knn_method"
                    )
                    k_neighbors = st.slider("Nombre de voisins (k)", 1, 20, 5, key="knn_k")

                    if st.button("Générer matrice", key="generate_knn_matrix"):
                        with st.spinner(f"Génération de la matrice de confusion KNN..."):
                            X_emb = embeddings_dict[selected_method]
                            fig_cm, accuracy = create_knn_confusion_matrix(X_emb, y, class_names, k_neighbors)
                            st.session_state.fig_cm = fig_cm
                            st.session_state.knn_accuracy = accuracy

                with col2:
                    if 'fig_cm' in st.session_state:
                        st.pyplot(st.session_state.fig_cm)
                        if 'knn_accuracy' in st.session_state:
                            st.metric("Précision KNN", f"{st.session_state.knn_accuracy:.3f}")

        # Tab 4: CSV Export
        with tab4:
            st.markdown("#### Export CSV des Embeddings")

            if embeddings_dict:
                st.info("Exporter tous les embeddings calculés dans un fichier CSV.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Générer CSV", key="generate_csv"):
                        csv_content = export_embeddings_csv(embeddings_dict, y, class_names)
                        st.session_state.csv_content = csv_content
                        st.success("CSV généré avec succès!")

                with col2:
                    if 'csv_content' in st.session_state:
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=st.session_state.csv_content,
                            file_name=f"embeddings_{meta.get('name', 'dataset')}.csv",
                            mime="text/csv"
                        )

                # Preview of CSV structure
                if 'csv_content' in st.session_state:
                    st.markdown("**Aperçu du fichier CSV:**")
                    df_preview = pd.read_csv(io.StringIO(st.session_state.csv_content))
                    st.dataframe(df_preview.head(10), use_container_width=True)
            else:
                st.warning("Aucun embedding disponible pour l'export.")

        # Tab 5: 3D Relief for Digits
        with tab5:
            st.markdown("#### Visualisation Relief 3D (Dataset Digits)")

            col1, col2 = st.columns([1, 2])
            with col1:
                if len(np.unique(y)) <= 10:  # Likely digits dataset
                    selected_digit = st.selectbox(
                        "Chiffre à visualiser:",
                        list(range(len(np.unique(y)))),
                        key="selected_digit"
                    )

                    if st.button("Générer relief 3D", key="generate_relief"):
                        with st.spinner("Génération du relief 3D..."):
                            fig_relief, error = create_digits_3d_relief(X, y, class_names, selected_digit)
                            if fig_relief is not None:
                                st.session_state.fig_relief = fig_relief
                            else:
                                st.error(error)
                else:
                    st.warning("Cette visualisation est optimisée pour le dataset Digits.")

            with col2:
                if 'fig_relief' in st.session_state:
                    st.plotly_chart(st.session_state.fig_relief, use_container_width=True)


def main():
    """Fonction principale de l'application."""
    seed_everything(42)
    initialize_session_state()

    # En-tête
    render_header()

    # Sélection du mode
    mode = render_sidebar_mode()

    # Gestion des données
    data_manager = DataManager()
    X, y, class_names, meta = render_data_section(data_manager)

    # Affichage selon le mode
    if mode == "🔬 Expérimenter un modèle":
        render_experiment_mode(X, y, class_names, meta, data_manager)
    else:
        render_compare_mode(X, y, class_names, meta, data_manager)

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ À propos"):
        st.markdown("""
        **Dimensionality Lab** - Laboratoire interactif de réduction de dimension

        - **Mode Expérimenter**: Testez un modèle spécifique avec sweep d'hyperparamètres
        - **Mode Comparer**: Comparez plusieurs modèles avec paramètres optimisés
        - **Datasets custom**: Placez vos fichiers dans le dossier `data/`

        Métriques calculées: Trustworthiness, Continuity, KNN Accuracy, Shepard correlation,
        Silhouette score, ARI, NMI, Stress
        """)

    return None  # Retour explicite pour éviter DeltaGenerator


if __name__ == "__main__":
    main()