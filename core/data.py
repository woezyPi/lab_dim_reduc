"""
Gestionnaire de données pour UMAP vs t-SNE Explorer
===================================================

Gère le chargement et le préprocessing des datasets :
- Digits (MNIST) : StandardScaler
- 20 Newsgroups : TfidfVectorizer → TruncatedSVD (sans densification)
- Génération de données synthétiques en fallback
- Cache Streamlit pour éviter les rechargements
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import warnings
import os
from pathlib import Path

from .utils import seed_everything

class DataManager:
    """Gestionnaire centralisé pour le chargement et preprocessing des données."""
    
    def __init__(self):
        """Initialise le gestionnaire de données."""
        seed_everything(42)
    
    @st.cache_data
    def load_digits(_self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Charge le dataset Digits (MNIST).
        
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Features normalisées
        y : ndarray of shape (n_samples,)
            Labels
        class_names : List[str]
            Noms des classes
        """
        X, y = load_digits(return_X_y=True)
        
        # Normalisation standard
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        class_names = [str(i) for i in range(10)]
        
        return X, y, class_names
    
    @st.cache_data  
    def load_newsgroups(_self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Charge le dataset 20 Newsgroups avec preprocessing TF-IDF → SVD.
        
        Returns
        -------
        X : ndarray of shape (n_samples, n_svd_components)
            Features TF-IDF réduites par SVD et normalisées
        y : ndarray of shape (n_samples,)
            Labels des catégories
        class_names : List[str]
            Noms des catégories
        """
        try:
            # Sélection de catégories équilibrées
            categories = [
                'alt.atheism',
                'comp.graphics', 
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'sci.crypt',
                'sci.electronics',
                'sci.med',
                'sci.space',
                'soc.religion.christian',
                'talk.politics.guns',
                'talk.politics.mideast',
                'talk.politics.misc'
            ]
            
            # Chargement avec nettoyage
            newsgroups = fetch_20newsgroups(
                subset='train',
                categories=categories,
                remove=('headers', 'footers', 'quotes'),
                random_state=42
            )
            
            # Vectorisation TF-IDF performante
            vectorizer = TfidfVectorizer(
                max_df=0.5,           # Ignore terms dans >50% des docs
                min_df=10,            # Ignore terms dans <10 docs
                max_features=50000,   # Limite le vocabulaire
                stop_words='english', # Supprime mots vides
                ngram_range=(1, 2),   # Unigrams + bigrams
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Mots alphabétiques uniquement
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Fit du vectorizer et transformation (garde sparse!)
            X_tfidf = vectorizer.fit_transform(newsgroups.data)
            
            # IMPORTANT: Ne jamais faire .toarray() sur X_tfidf!
            # Utilisation de TruncatedSVD directement sur la matrice sparse
            svd = TruncatedSVD(
                n_components=300,
                algorithm='randomized',
                n_iter=5,
                random_state=42
            )
            
            X_svd = svd.fit_transform(X_tfidf)
            
            # Normalisation finale
            scaler = StandardScaler()
            X = scaler.fit_transform(X_svd)
            
            # Nettoyage des noms de catégories
            class_names = [name.split('.')[-1].title() for name in newsgroups.target_names]
            
            return X, newsgroups.target, class_names
            
        except Exception as e:
            warnings.warn(f"Échec du téléchargement 20 Newsgroups: {str(e)}")
            st.warning(f"⚠️ Impossible de télécharger 20 Newsgroups: {str(e)}")
            st.info("🔄 Génération de données textuelles synthétiques...")
            
            return self._generate_synthetic_text_data()
    
    def _generate_synthetic_text_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Génère des données synthétiques simulant un dataset textuel après TF-IDF → SVD.
        
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Features synthétiques normalisées
        y : ndarray of shape (n_samples,)
            Labels synthétiques
        class_names : List[str]
            Noms des catégories synthétiques
        """
        np.random.seed(42)
        
        # Configuration
        n_categories = 8
        n_samples_per_cat = 250
        n_features = 300  # Simulation post-SVD
        
        class_names = [
            'Technology', 'Sports', 'Politics', 'Science',
            'Health', 'Entertainment', 'Business', 'Education'
        ]
        
        X_list = []
        y_list = []
        
        for cat_idx in range(n_categories):
            # Pattern spécifique par catégorie (simule l'effet SVD)
            
            # Composantes principales dominantes pour cette catégorie
            dominant_components = np.random.choice(n_features, size=20, replace=False)
            
            # Base gaussienne centrée
            category_data = np.random.normal(0, 0.5, (n_samples_per_cat, n_features))
            
            # Amplification des composantes dominantes
            for comp in dominant_components:
                category_data[:, comp] += np.random.normal(
                    loc=2.0 * (cat_idx + 1) / n_categories,  # Signal par catégorie
                    scale=0.3,
                    size=n_samples_per_cat
                )
            
            # Corrélations inter-features (simule structure thématique)
            correlation_matrix = np.eye(n_features)
            for i in range(0, n_features, 10):
                end = min(i + 10, n_features)
                correlation_matrix[i:end, i:end] = 0.3
            
            # Transformation corrélée
            L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(n_features))
            category_data = category_data @ L.T
            
            X_list.append(category_data)
            y_list.extend([cat_idx] * n_samples_per_cat)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Normalisation finale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Mélange reproductible
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        return X, y, class_names
    
    @st.cache_resource
    def get_vectorizer_and_svd(_self, max_features: int = 50000, 
                               svd_components: int = 300) -> Tuple[TfidfVectorizer, TruncatedSVD]:
        """
        Crée et retourne des objets vectorizer et SVD configurés.
        
        Parameters
        ----------
        max_features : int, default=50000
            Nombre maximum de features TF-IDF
        svd_components : int, default=300
            Nombre de composantes SVD
            
        Returns
        -------
        vectorizer : TfidfVectorizer
            Vectorizer configuré
        svd : TruncatedSVD
            Réducteur SVD configuré
        """
        vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=10,
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
            lowercase=True,
            strip_accents='unicode'
        )
        
        svd = TruncatedSVD(
            n_components=svd_components,
            algorithm='randomized', 
            n_iter=5,
            random_state=42
        )
        
        return vectorizer, svd
    
    def preprocess_text_pipeline(self, texts: List[str], 
                                svd_components: int = 300) -> np.ndarray:
        """
        Pipeline complète de preprocessing pour texte: TF-IDF → SVD → StandardScaler.
        
        Parameters
        ----------
        texts : List[str]
            Textes à préprocesser
        svd_components : int, default=300
            Nombre de composantes SVD
            
        Returns
        -------
        X : ndarray of shape (n_samples, svd_components)
            Features normalisées prêtes pour embedding
        """
        vectorizer, svd = self.get_vectorizer_and_svd(svd_components=svd_components)
        
        # TF-IDF (garde sparse)
        X_tfidf = vectorizer.fit_transform(texts)
        
        # SVD sur sparse
        X_svd = svd.fit_transform(X_tfidf)
        
        # Normalisation
        scaler = StandardScaler()
        X = scaler.fit_transform(X_svd)
        
        return X
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Retourne les informations sur un dataset.
        
        Parameters
        ----------
        dataset_name : str
            Nom du dataset ('digits' ou 'newsgroups')
            
        Returns
        -------
        info : dict
            Informations sur le dataset
        """
        if dataset_name.lower() in ['digits', 'mnist']:
            return {
                'name': 'Digits (MNIST)',
                'type': 'image',
                'n_classes': 10,
                'recommended_metric': 'euclidean',
                'preprocessing': 'StandardScaler',
                'description': 'Chiffres manuscrits 8x8 pixels'
            }
        elif dataset_name.lower() in ['newsgroups', '20newsgroups', 'text']:
            return {
                'name': '20 Newsgroups',
                'type': 'text', 
                'n_classes': 15,
                'recommended_metric': 'cosine',
                'preprocessing': 'TF-IDF → SVD → StandardScaler',
                'description': 'Textes de forums thématiques'
            }
        else:
            return {
                'name': 'Unknown',
                'type': 'unknown',
                'n_classes': 0,
                'recommended_metric': 'euclidean',
                'preprocessing': 'None',
                'description': 'Dataset non reconnu'
            }

    def load_train_test_split(self, train_path: str, test_path: Optional[str] = None,
                            has_labels: bool = True) -> Dict[str, Any]:
        """
        Charge un dataset avec split train/test.

        Parameters
        ----------
        train_path : str
            Chemin vers le fichier train
        test_path : str, optional
            Chemin vers le fichier test
        has_labels : bool
            Si True, la dernière colonne contient les labels

        Returns
        -------
        data : Dict
            Dictionnaire contenant X_train, y_train, X_test, y_test (si fourni)
        """
        data = {}

        # Charger le train
        if train_path.endswith('.csv'):
            df_train = pd.read_csv(train_path)
            if has_labels:
                X_train = df_train.iloc[:, :-1].values
                y_train = df_train.iloc[:, -1].values
            else:
                X_train = df_train.values
                y_train = np.zeros(len(X_train))
        elif train_path.endswith('.npy'):
            X_train = np.load(train_path)
            y_train = np.zeros(len(X_train))  # Labels par défaut

        data['X_train'] = X_train
        data['y_train'] = y_train

        # Charger le test si fourni
        if test_path:
            if test_path.endswith('.csv'):
                df_test = pd.read_csv(test_path)
                if has_labels:
                    X_test = df_test.iloc[:, :-1].values
                    y_test = df_test.iloc[:, -1].values
                else:
                    X_test = df_test.values
                    y_test = np.zeros(len(X_test))
            elif test_path.endswith('.npy'):
                X_test = np.load(test_path)
                y_test = np.zeros(len(X_test))

            data['X_test'] = X_test
            data['y_test'] = y_test

        return data

    def list_datasets(self, data_dir: str = "data") -> List[str]:
        """
        Liste les datasets disponibles dans le dossier data/.

        Parameters
        ----------
        data_dir : str, default="data"
            Dossier contenant les datasets

        Returns
        -------
        datasets : List[str]
            Liste des noms de datasets trouvés
        """
        datasets = []
        data_path = Path(data_dir)

        if data_path.exists():
            # Chercher les sous-dossiers et fichiers
            for item in data_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Vérifier si contient des données
                    has_data = any(
                        list(item.glob("*.csv")) +
                        list(item.glob("*.npy")) +
                        list(item.glob("*.npz")) +
                        list(item.glob("*.parquet"))
                    )
                    if has_data:
                        datasets.append(item.name)
                elif item.is_file():
                    if item.suffix in ['.csv', '.npy', '.npz', '.parquet']:
                        datasets.append(item.stem)

        return datasets

    @st.cache_data
    def load_dataset(_self, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Charge un dataset par son nom.

        Parameters
        ----------
        name : str
            Nom du dataset

        Returns
        -------
        X : ndarray
            Features
        y : ndarray
            Labels (ou indices si non supervisé)
        class_names : List[str]
            Noms des classes
        meta : dict
            Métadonnées du dataset
        """
        # Datasets built-in
        if name in ["Digits (MNIST)", "Digits", "MNIST"]:
            X, y, class_names = _self.load_digits()
            meta = {
                "name": "Digits (MNIST)",
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(class_names),
                "type": "image"
            }
            return X, y, class_names, meta

        elif name in ["20 Newsgroups", "Newsgroups", "Text"]:
            X, y, class_names = _self.load_newsgroups()
            meta = {
                "name": "20 Newsgroups",
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(class_names),
                "type": "text"
            }
            return X, y, class_names, meta

        # Datasets custom dans data/
        data_path = Path("data")

        # Chercher dans un sous-dossier
        dataset_dir = data_path / name
        if dataset_dir.exists() and dataset_dir.is_dir():
            return _self._load_from_directory(dataset_dir, name)

        # Chercher un fichier direct
        for ext in ['.csv', '.npy', '.npz', '.parquet']:
            file_path = data_path / f"{name}{ext}"
            if file_path.exists():
                return _self._load_from_file(file_path, name)

        raise ValueError(f"Dataset '{name}' not found")

    def _load_from_directory(self, dir_path: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Charge un dataset depuis un dossier."""
        # Chercher les fichiers de données
        csv_files = list(dir_path.glob("*.csv"))
        npy_files = list(dir_path.glob("*.npy"))
        npz_files = list(dir_path.glob("*.npz"))
        parquet_files = list(dir_path.glob("*.parquet"))

        if csv_files:
            return self._load_csv(csv_files[0], name)
        elif npz_files:
            return self._load_npz(npz_files[0], name)
        elif npy_files:
            # Chercher X.npy et y.npy
            x_file = dir_path / "X.npy"
            y_file = dir_path / "y.npy"
            if x_file.exists():
                X = np.load(x_file)
                y = np.load(y_file) if y_file.exists() else np.zeros(len(X))
                return self._create_dataset_meta(X, y, name)
        elif parquet_files:
            return self._load_parquet(parquet_files[0], name)

        raise ValueError(f"No valid data files found in {dir_path}")

    def _load_from_file(self, file_path: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Charge un dataset depuis un fichier."""
        if file_path.suffix == '.csv':
            return self._load_csv(file_path, name)
        elif file_path.suffix == '.npz':
            return self._load_npz(file_path, name)
        elif file_path.suffix == '.npy':
            X = np.load(file_path)
            y = np.zeros(len(X))  # Non supervisé
            return self._create_dataset_meta(X, y, name)
        elif file_path.suffix == '.parquet':
            return self._load_parquet(file_path, name)

        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_csv(self, file_path: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Charge un fichier CSV."""
        df = pd.read_csv(file_path)

        # Détection du type de dataset
        # Chercher la colonne "Class Index" ou la première colonne numérique comme labels
        y = None
        class_col = None

        # Chercher une colonne de classe
        for col in ['Class Index', 'class', 'label', 'target', 'category']:
            if col in df.columns:
                class_col = col
                break

        # Si on a trouvé une colonne de classe, l'extraire
        if class_col:
            y_raw = df[class_col]
            # Encoder les labels si nécessaire
            if y_raw.dtype == 'object':
                unique_labels = y_raw.unique()
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = y_raw.map(label_map).values
                class_names = [str(label) for label in unique_labels]
            else:
                y = y_raw.values.astype(int)
                unique_labels = np.unique(y)
                class_names = [f"Class {i}" for i in unique_labels]

            # Retirer la colonne de classe du DataFrame
            df = df.drop(columns=[class_col])
        else:
            # Pas de labels trouvés
            y = np.zeros(len(df))
            class_names = ["Cluster 0"]

        # Détection des colonnes textuelles
        text_columns = []
        numeric_columns = []

        for col in df.columns:
            if df[col].dtype == 'object':
                text_columns.append(col)
            else:
                numeric_columns.append(col)

        # Traitement selon le type de données
        if text_columns:
            # Dataset textuel - utiliser TF-IDF et SVD
            if len(text_columns) == 1:
                # Une seule colonne texte
                texts = df[text_columns[0]].fillna('').astype(str).tolist()
            else:
                # Concatener les colonnes texte
                texts = df[text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

            # Pipeline TF-IDF → SVD
            X = self.preprocess_text_pipeline(texts, svd_components=min(300, len(texts) - 1))
            dtype = "text"

        elif numeric_columns:
            # Dataset numérique
            X = df[numeric_columns].values.astype(np.float32)

            # Normalisation
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Détection du type
            if X.shape[1] == 64:
                dtype = "image"
            else:
                dtype = "tabulaire"
        else:
            raise ValueError(f"No valid data columns found in {file_path}")

        meta = {
            "name": name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "type": dtype
        }

        return X, y, class_names, meta

    def _load_npz(self, file_path: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Charge un fichier NPZ."""
        data = np.load(file_path)

        X = data['X'] if 'X' in data else data['x'] if 'x' in data else data['data']
        y = data['y'] if 'y' in data else data['labels'] if 'labels' in data else np.zeros(len(X))

        if sparse.issparse(X):
            X = X.toarray()

        # Normalisation
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        unique_labels = np.unique(y)
        class_names = [f"Class {i}" for i in unique_labels]

        return self._create_dataset_meta(X, y.astype(int), name)

    def _load_parquet(self, file_path: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Charge un fichier Parquet."""
        df = pd.read_parquet(file_path)
        # Même logique que CSV
        return self._load_csv(file_path, name)  # Réutilise la logique CSV

    def _create_dataset_meta(self, X: np.ndarray, y: np.ndarray,
                            name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Crée les métadonnées pour un dataset."""
        unique_labels = np.unique(y)
        class_names = [f"Class {i}" for i in unique_labels]

        # Détection du type
        if X.shape[1] == 64:  # Probablement des images 8x8
            dtype = "image"
        elif X.shape[1] > 1000:  # Probablement du texte (TF-IDF)
            dtype = "text"
        else:
            dtype = "tabulaire"

        meta = {
            "name": name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(unique_labels),
            "type": dtype
        }

        return X, y, class_names, meta

    @staticmethod
    def describe_dataset(meta: Dict[str, Any]) -> str:
        """
        Génère une description textuelle d'un dataset.

        Parameters
        ----------
        meta : dict
            Métadonnées du dataset

        Returns
        -------
        description : str
            Description formatée
        """
        return (f"📊 {meta['name']}: {meta['n_samples']} échantillons, "
                f"{meta['n_features']} features, {meta['n_classes']} classes ({meta['type']})")