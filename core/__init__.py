#!/usr/bin/env python3
"""
Script de configuration de l'environnement
===========================================

Installe les dépendances et configure l'environnement pour l'application.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Exécute une commande avec gestion d'erreur."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} terminé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de {description}")
        print(f"Commande: {cmd}")
        print(f"Erreur: {e.stderr}")
        return False

def check_python_version():
    """Vérifie la version Python."""
    version = sys.version_info
    print(f"🐍 Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ requis")
        return False
    
    print("✅ Version Python compatible")
    return True

def create_directories():
    """Crée les répertoires nécessaires."""
    directories = [
        'core',
        'tests', 
        'data',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Répertoire {directory}/ créé")
    
    return True

def install_dependencies():
    """Installe les dépendances Python."""
    if not Path('requirements.txt').exists():
        print("❌ Fichier requirements.txt introuvable")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installation des dépendances"
    )

def install_dev_dependencies():
    """Installe les dépendances de développement."""
    dev_packages = [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0', 
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0'
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(dev_packages)}"
    return run_command(cmd, "Installation des outils de développement")

def verify_installation():
    """Vérifie que les packages critiques sont installés."""
    critical_packages = [
        'streamlit',
        'numpy', 
        'pandas',
        'scikit-learn',
        'matplotlib',
        'umap-learn',
        'hdbscan'
    ]
    
    print("🔍 Vérification des packages critiques...")
    
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} non trouvé")
            return False
    
    return True

def create_sample_config():
    """Crée un fichier de configuration d'exemple."""
    config_content = """# Configuration UMAP vs t-SNE Explorer
# ====================================

# Paramètres par défaut
DEFAULT_DATASET = "Digits (MNIST)"
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_EXECUTION_MODE = "Rapide"

# Paramètres UMAP par défaut
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SPREAD = 1.0

# Paramètres t-SNE par défaut  
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = "auto"
TSNE_MAX_ITER = 1000

# Interface
USE_PLOTLY = false
PLOT_SIZE = 6
POINT_SIZE = 10
POINT_ALPHA = 0.7

# Performance
N_JOBS = 1
CACHE_SIZE = "1GB"
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("📝 Fichier config.py créé")
    return True

def main():
    """Configuration principale."""
    print("🚀 Configuration de l'environnement UMAP vs t-SNE Explorer")
    print("=" * 60)
    
    steps = [
        ("Vérification version Python", check_python_version),
        ("Création des répertoires", create_directories), 
        ("Installation des dépendances", install_dependencies),
        ("Installation outils de développement", install_dev_dependencies),
        ("Vérification de l'installation", verify_installation),
        ("Création configuration d'exemple", create_sample_config)
    ]
    
    success_count = 0
    
    for description, func in steps:
        print(f"\n📋 {description}")
        if func():
            success_count += 1
        else:
            print(f"❌ Échec: {description}")
    
    print(f"\n📊 Résultat: {success_count}/{len(steps)} étapes réussies")
    
    if success_count == len(steps):
        print("\n🎉 Configuration terminée avec succès !")
        print("\n🚀 Pour lancer l'application:")
        print("   streamlit run app.py")
        print("\n🧪 Pour lancer les tests:")
        print("   python run_tests.py")
        return 0
    else:
        print("\n❌ Configuration incomplète")
        print("Vérifiez les erreurs ci-dessus")
        return 1

if __name__ == "__main__":
    sys.exit(main())