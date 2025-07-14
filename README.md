# 📄 Matching Offres-Profils IA

Une application Streamlit complète pour :

- 🔍 Analyser des offres d’emploi (txt/csv) via extraction de compétences
- 🤝 Faire du matching intelligent entre profils et offres (via NLP, ML ou DL)
- 🧠 Entraîner et tester des modèles Machine Learning ou Deep Learning
- 📈 Comparer les performances des modèles ML vs DL
- 💾 Exporter les modèles et résultats

---

## 🚀 Fonctionnalités principales

### 1. Analyse d'Offres
- Extraction des compétences à partir des colonnes sélectionnées
- Détection de la localisation (ville, région, département) et du niveau d'expérience
- Résumé par offre avec aperçu des éléments extraits

### 2. Matching Complet
- Matching entre les offres extraites et les profils en base
- Filtrage par nombre minimum de compétences
- Visualisation : diagrammes en secteurs (Pie chart), barres (bar chart), statistiques par niveau de matching
- Export CSV des résultats

### 3. Machine Learning
- Entraînement de modèles : Random Forest, SVM, Logistic Regression
- Prédiction intelligente à partir d'une offre
- Export du modèle entraîné au format `.pkl`

### 4. Deep Learning
- Matching par réseau de neurones via `MatchingDLTrainer`
- Entraînement, évaluation et prédiction
- Utilisation d’embeddings NLP (spaCy + Transformers)

### 5. Comparaison ML / DL
- Affichage côte à côte des performances des modèles
- Graphiques et recommandations d’usage

---

## 📦 Installation

*(exemple d'installation si vous voulez)*  
```bash
pip install -r requirements.txt
streamlit run app.py
