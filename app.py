import datetime
import os
import pickle
import time
import traceback
import joblib
import streamlit as st
import pandas as pd
import spacy
from datetime import datetime  # Ajoutez cette ligne avec les autres imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from torch import cosine_similarity
import torch
import torch.nn.functional as F

# Exemple de calcul de similarité cosinus

from database import init_db_connection
from fonction import safe_extract, safe_lower
import plotly.express as px  # Ajoutez cette ligne avec les autres imports
from keywords_localisation import get_profile_keywords_from_postgres, normalize_city_name
from sqlalchemy import create_engine
from filtre import extract_filters
from matching import create_matching_stats_chart, filter_and_rank_matches, match_all_offers_to_profiles
from matching_dl import MatchingDLTrainer
from matching_ml import MatchingTrainer
from offres import extract_offre_info
from preparation_data import display_predictions, prepare_training_data
from profile_1 import load_profiles
import re  # Ajoutez cette ligne avec les autres imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from run_dl import run_deep_learning
# --- Setup (à adapter à ta config) ---*
@st.cache_resource(show_spinner=False)
def get_engine():
    db = st.session_state['db_params']
    url = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    return create_engine(url, connect_args={
        'client_encoding': 'latin1',  # or 'WIN1252' if needed
        'options': '-c client_encoding=latin1'
    })
if not init_db_connection():
    st.warning("Veuillez configurer la connexion à la base de données")
    st.stop()
@st.cache_resource(show_spinner=False)
def get_engine():
    db = st.session_state['db_params']
    url = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    return create_engine(url)

engine = get_engine()

# Charger modèle NLP
@st.cache_resource(show_spinner=False)
def load_nlp():
    return spacy.load("fr_core_news_sm")

nlp = load_nlp()

#st.title("✅ Test chargement Profils & Extraction Offre")

# --- Charger profils ---
profils_df = load_profiles()
if profils_df is None or len(profils_df) == 0:
    st.error("Aucun profil trouvé dans la base de données")
   
    st.stop()
#st.write(profils_df.head())
# --- Charger mots-clés profils pour extraction offre ---
profile_keywords = get_profile_keywords_from_postgres()
#st.write(f"Nombre de mots-clés extraits de la base : {len(profile_keywords)}")
# Après le chargement des profils
if not profils_df.empty:
    # Création des ensembles de référence
    villes_reference = frozenset(profils_df['localisation'].str.lower().dropna().unique())
    regions_reference = frozenset(profils_df['region'].str.lower().dropna().unique())
    departments_reference = frozenset(profils_df['department'].str.lower().dropna().unique())
else:
    st.warning("Aucun profil chargé pour la référence des localisations")
with st.sidebar:
   
    choice = st.sidebar.radio("Navigation", ["Analyse Offre",  "Match Complet", "Machine learning", "Deep Learning","Test Model Extracte ML",  "Comparaison ML/DL"])   

# --- Upload offre ---
uploaded_file = st.sidebar.file_uploader("Charger une offre (.txt ou .csv)", type=["txt", "csv"])
if uploaded_file is not None:
    df_offres= pd.read_csv(
        uploaded_file,
        sep=';',
        encoding='latin1',
        on_bad_lines='warn'
    )
    #st.write(f"Nombre d'offres chargées : {len(df_offres)}")
else:
    st.write("Aucune offre chargée pour le moment.")
    df_offres = None    

if "offres" not in st.session_state:
    st.session_state.offres = []

# Téléchargement du fichier
import json  # Ajout de l'import manquant
import pandas as pd
import streamlit as st
from filtre import extract_filters
from keywords_localisation import get_profile_keywords_from_postgres
import spacy

# Charger le modèle spaCy
nlp = spacy.load("fr_core_news_sm")

if choice == "Analyse Offre":
    st.title("📄 Analyse d'Offres avec Détection par Mot-clé")
    
    try:
        uploaded_file.seek(0)
        df_offres = pd.read_csv(uploaded_file, sep=';', encoding='latin1', on_bad_lines='warn')
        
        if df_offres.empty:
            st.error("Le fichier est vide ou corrompu")
            st.stop()

        # Nettoyage initial
        df_offres = df_offres.dropna(how='all').fillna('')
        st.success(f"✅ {len(df_offres)} offres valides après filtrage")
        
        # Sélection des colonnes
        st.subheader("🔍 Sélection des données sources")
        selected_cols = st.multiselect(
            "Colonnes à analyser",
            options=df_offres.columns,
            default=list(df_offres.columns)[:2]
        )

        if not selected_cols:
            st.warning("Veuillez sélectionner au moins une colonne")
            st.stop()

        if st.button("🔎 Détecter les compétences", type="primary"):
            with st.spinner("Analyse en cours..."):
                progress = st.progress(0)
                total = len(df_offres)

                results = []
                profile_keywords = get_profile_keywords_from_postgres()
                
                for i, row in df_offres.iterrows():
                    try:
                        # Combinaison du texte
                        full_text = " ".join(str(row[col]) for col in selected_cols).strip()
                        
                        if not full_text:
                            continue
                            
                        # Détection des compétences
                        doc = nlp(full_text.lower())
                        detected_skills = {
                            token.text for token in doc 
                            if token.text in profile_keywords and len(token.text) > 2
                        }
                        
                        if not detected_skills:
                            continue
                            
                        # Extraction des métadonnées
                        filters = extract_filters(full_text, villes_reference, regions_reference, departments_reference)
                        
                        results.append({
                            'Offre': f"Offre {i+1}",
                            'Extrait': full_text[:200] + "..." if len(full_text) > 200 else full_text,
                            'Compétences': sorted(list(detected_skills)),
                            'Localisation': filters.get('localisation', 'Non détectée'),
                            'Expérience': filters.get('experience_level', 'Non spécifiée')
                        })
                        progress.progress((i + 1) / total)
                
                    except Exception as e:
                        st.warning(f"Ignorée ligne {i+1}: {str(e)}")
                        continue

                if not results:
                    st.warning("Aucune compétence détectée dans les offres")
                    st.stop()
                
                st.session_state.offres = results
                
                # Affichage des résultats
                st.subheader(f"🔎 {len(results)} offres avec compétences")
              
                # Tableau synthétique
                df_results = pd.DataFrame(results)
                st.dataframe(df_results[['Offre', 'Localisation', 'Expérience']])
                
                # Détails par offre
                selected = st.selectbox(
                    "Voir le détail pour:",
                    options=range(len(results)),
                    format_func=lambda x: results[x]['Offre']
                )
                
                with st.expander("📝 Détails complets"):
                    st.write("**Texte complet:**")
                    st.write(results[selected]['Extrait'])
                    
                    st.write("**Compétences détectées:**")
                    for skill in results[selected]['Compétences']:
                        st.markdown(f"- {skill}")
                
                # Export JSON
                json_data = json.dumps(results, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 Télécharger les résultats",
                    data=json_data,
                    file_name="analyse_offres.json",
                    mime="application/json"
                )

    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        st.stop()
elif choice == "Match Complet":
    st.title("🤝 Matching Complet Offres-Profils")
    
    # Paramètres
    min_skills = st.sidebar.slider(
        "Compétences minimum requises",
        min_value=1, max_value=5, value=2,
        help="Commencez par 2 et augmentez si trop de résultats"
    )
    
    if st.button("🔍 Lancer le Matching", type="primary"):
        # Conversion des offres
        with st.spinner("Préparation des données..."):
            offers_for_matching = [
                {
                    'Offre': f"Offre {i+1}",
                    'competences': offre['Compétences'],
                    'localisation': offre.get('Localisation', []),
                    'experience_level': str(offre.get('Expérience', '')).lower()
                }
                for i, offre in enumerate(st.session_state.offres)
            ]
        
        # Matching avec barre de progression
        try:
            matches_df = match_all_offers_to_profiles(
                offers_for_matching,
                profils_df,
                min_skills=min_skills
            )
            
            if matches_df.empty:
                st.error("Aucune correspondance trouvée. Conseils:")
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Vérifiez:**")
                    st.write("- Compétences minimum")
                    st.write("- Formats des localisations")
                with cols[1]:
                    st.write("**Debug:**")
                    st.write(f"Offres: {len(offers_for_matching)}")
                    st.write(f"Profils: {len(profils_df)}")
                st.stop()
            
            # Classement et affichage
            ranked_matches = filter_and_rank_matches(matches_df)
            st.session_state.matches = ranked_matches
            
            # Nouvelle visualisation par label
            st.subheader("📊 Répartition des Matchs par Niveau")
            col1, col2 = st.columns(2)
            
            with col1:
                # Diagramme en camembert
                label_counts = ranked_matches['label'].value_counts().sort_index()
                fig1 = px.pie(
                    names=[f"Niveau {i}" for i in label_counts.index],
                    values=label_counts.values,
                    title="Distribution des Niveaux de Matching",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Diagramme en barres
                label_stats = ranked_matches.groupby('label').agg({
                    'nb_common_skills': 'mean',
                    'location_match': 'mean',
                    'experience_match': 'mean'
                }).reset_index()
                
                fig2 = px.bar(
                    label_stats,
                    x='label',
                    y=['nb_common_skills', 'location_match', 'experience_match'],
                    title="Métriques par Niveau de Matching",
                    labels={'value': 'Valeur moyenne', 'variable': 'Métrique'},
                    barmode='group'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Affichage organisé avec onglets
            tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Détails par Niveau", "Statistiques", "Export"])
            
            with tab1:
                # Vérifiez d'abord si la colonne existe
                columns_to_show = [
                    'offer_title', 'profile_poste', 'nb_common_skills',
                    'label', 'score'
                ]
                
                # Ajoutez 'matching_reason' seulement si elle existe
                # Ajoutez cette ligne après avoir obtenu ranked_matches
                if 'matching_reason' not in ranked_matches.columns:
                    ranked_matches['matching_reason'] = "Non spécifié"
                    columns_to_show.append('matching_reason')
                
                st.dataframe(
                    ranked_matches.head(100)[columns_to_show].rename(columns={
                        'offer_title': 'Offre',
                        'profile_poste': 'Profil',
                        'nb_common_skills': 'Compétences',
                        'label': 'Niveau',
                        'score': 'Score',
                        'matching_reason': 'Raison du Matching'
                    }),
                    height=500
                )
            
            with tab2:
                selected_label = st.selectbox(
                    "Sélectionnez un niveau de matching à analyser",
                    options=sorted(ranked_matches['label'].unique()),
                    format_func=lambda x: f"Niveau {x}"
                )
                
                filtered = ranked_matches[ranked_matches['label'] == selected_label]
                st.write(f"### 🔍 {len(filtered)} matchs de niveau {selected_label}")
                
                # Colonnes à afficher avec vérification
                columns_to_display = [
                    'offer_title', 'profile_poste', 'nb_common_skills',
                    'location_match', 'experience_match'
                ]
                
                if 'matching_reason' in filtered.columns:
                    columns_to_display.append('matching_reason')
                
                st.dataframe(filtered[columns_to_display])
            
            with tab3:
                st.plotly_chart(create_matching_stats_chart(ranked_matches))
                
                st.write("### 🔢 Statistiques Globales")
                cols = st.columns(3)
                cols[0].metric("Total Matchs", len(ranked_matches))
                cols[1].metric("Matchs Complets (N3)", 
                              len(ranked_matches[ranked_matches['label'] == 3]))
                cols[2].metric("Compétences Moyennes", 
                              f"{ranked_matches['nb_common_skills'].mean():.1f}")
                
                st.write("### 📍 Répartition Géographique")
                loc_counts = ranked_matches['profile_location'].value_counts().head(10)
                st.bar_chart(loc_counts)
            
            with tab4:
                st.download_button(
                    "📤 Exporter CSV complet",
                    ranked_matches.to_csv(index=False, encoding='utf-8-sig'),
                    f"matching_complet.csv",
                    "text/csv"
                )
                st.download_button(
                    "📊 Exporter Top 100",
                    ranked_matches.head(100).to_csv(index=False, encoding='utf-8-sig'),
                    f"top100_matching.csv",
                    "text/csv"
                )
                st.download_button(
                    "📈 Exporter Statistiques",
                    label_stats.to_csv(index=False, encoding='utf-8-sig'),
                    "stats_matching.csv",
                    "text/csv"
                )
        
        except Exception as e:
            st.error(f"Erreur lors du matching: {str(e)}")
            st.write("Vérifiez les données en entrée via l'onglet Debug")
    if 'historical_matches' not in st.session_state:
        st.session_state.historical_matches = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
# 
elif choice == "Machine learning":
    st.title("🤖 Matching Intelligent avec ML")

    # Sélection du modèle AVANT l'entraînement
    model_choice = st.selectbox(
        "Choisissez le modèle de ML",
        options=["random_forest", "logistic_regression", "svm"],
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "logistic_regression": "Régression Logistique",
            "svm": "SVM"
        }[x]
    )

    # Section Entraînement
    if st.button("🔄 Lancer l'entraînement") and 'matches' in st.session_state:
        with st.spinner("Entraînement en cours..."):
            X_train = [
                f"{row['offer_title']} {row['matched_skills']}" 
                for _, row in st.session_state.matches.iterrows()
            ]
            y_train = st.session_state.matches['label'].values

            # Passer le choix du modèle ici
            trainer = MatchingTrainer(model_type=model_choice)
            metrics = trainer.train(X_train, y_train)

            st.session_state.model = trainer
            st.session_state.metrics = metrics
            st.success("✅ Modèle entraîné avec succès !")
            st.json(metrics)

    # Section Prédiction et Export
    if 'model' in st.session_state:
        with st.expander("🔮 Prédire des correspondances", expanded=True):
            # offer_text = st.text_area(
            #     "Texte de l'offre",
            #     value="développeur fullskack java",
            #     height=100
            # )
            if "offer_text" not in st.session_state:
                st.session_state.offer_text = "développeur fullskack java"

            offer_text = st.text_area(
                "Texte de l'offre",
                value=st.session_state.offer_text,
                height=100
            )

            # Mettre à jour la session à chaque changement
            st.session_state.offer_text = offer_text


            if st.button("🔍 Trouver les meilleurs profils"):
                try:
                    with st.spinner("Recherche des profils correspondants..."):
                        predictions = st.session_state.model.predict(offer_text, profils_df)

                        display_predictions(predictions, profils_df, offer_text)

                        with st.expander("📊 Détails techniques"):
                            st.json({
                                "offre_analysée": offer_text,
                                "top_profils_ids": predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                            })
                except Exception as e:
                    st.error(f"Erreur de prédiction: {str(e)}")

        st.divider()
        st.subheader("📤 Exporter le modèle")

        model_name = st.text_input("Nom du modèle", f"{model_choice}_matching")

        if st.button("💾 Exporter le modèle"):
            try:
                filename = f"{model_name}.pkl"

                model_data = {
                    'model': st.session_state.model.model,
                    'metadata': {
                        'performance': st.session_state.metrics,
                        'model_type': model_choice
                    }
                }

                st.download_button(
                    label="Télécharger le modèle",
                    data=pickle.dumps(model_data),
                    file_name=filename,
                    mime="application/octet-stream"
                )
                st.success(f"Modèle exporté sous {filename}")
            except Exception as e:
                st.error(f"Erreur d'export: {str(e)}")
        st.warning("Veuillez d'abord entraîner le modèle")

elif choice == "Test Model Extracte ML":
    st.title("🧪 Tester un Modèle")
    
    # 1. Model Loading with enhanced validation
    uploaded_model = st.file_uploader("📤 Charger un modèle (.pkl)", type="pkl")
    
    if uploaded_model:
        try:
            with st.spinner("Chargement du modèle..."):
                model_data = pickle.load(uploaded_model)
                
                # Handle all possible model formats
                if isinstance(model_data, dict):
                    model = model_data.get('pipeline', model_data.get('model', None))
                else:
                    model = model_data
                
                if model is None:
                    raise ValueError("Format de modèle non reconnu")
                
                # Validate model structure
                if not hasattr(model, 'named_steps') or 'tfidfvectorizer' not in model.named_steps:
                    raise ValueError("Modèle invalide - structure de pipeline incorrecte")
                
                st.session_state.tested_model = model
                st.success("✅ Modèle chargé avec succès!")
                
                # Show model metadata
                with st.expander("🔍 Métadonnées du modèle"):
                    try:
                        st.json({
                            "model_type": str(type(model)),
                            "n_features": len(model.named_steps['tfidfvectorizer'].get_feature_names_out()),
                            "classes": model.named_steps['randomforestclassifier'].classes_.tolist()
                        })
                    except Exception as e:
                        st.warning(f"Metadata incomplète: {str(e)}")

        except Exception as e:
            st.error(f"Erreur de chargement: {str(e)}")
            st.stop()
    
    # 2. Model Testing with robust prediction handling
    if 'tested_model' in st.session_state:
        st.divider()
        st.subheader("🔎 Tester sur une nouvelle offre")
        try:
                        info = {
                            "type": str(type(model)),
                            "paramètres": model.get_params()
                        }
                        
                        if hasattr(model, 'named_steps'):
                            info["étapes"] = list(model.named_steps.keys())
                            info["n_features"] = len(model.named_steps['tfidfvectorizer'].get_feature_names_out())
                        
                        st.json(info)
        except Exception as e:
                        st.warning(f"Metadata partielle: {str(e)}")

        except Exception as e:
            st.error(f"Erreur de chargement: {str(e)}")
            st.error("Le fichier doit contenir un modèle scikit-learn ou un pipeline valide")
            st.stop()

        
        offer_text = st.text_area(
            "Saisir le texte de l'offre",
            "développeur fullstack java",
            height=150
        )
        
        if st.button("🔍 Trouver les profils correspondants"):
            if not offer_text.strip():
                st.warning("Veuillez saisir un texte d'offre valide")
                st.stop()
            
            # Initialize all variables
            predictions = None
            features = None
            
            try:
                model = st.session_state.tested_model
                
                # Feature extraction debug
                with st.expander("🛠 Debug Information"):
                    try:
                        vectorizer = model.named_steps['tfidfvectorizer']
                        features = vectorizer.transform([offer_text])
                        feature_names = vectorizer.get_feature_names_out()
                        st.write("Full vocabulary size:", len(feature_names))
                        st.write("Top features for this offer:")
                        top_features = sorted(zip(features.nonzero()[1], features.data), key=lambda x: -x[1])
                        for idx, score in top_features[:10]:
                            st.write(f"{feature_names[idx]}: {score:.4f}")
                        active_features = [
                            (int(idx), feature_names[idx], float(features[0, idx]))
                            for idx in features.nonzero()[1]
                        ]
                        
                        st.write("Feature shape:", features.shape)
                        st.write("Active features:", active_features)
                    except Exception as e:
                        st.error(f"Feature extraction failed: {str(e)}")
                        st.stop()
                
                # Make prediction
                try:
                    with st.spinner("Prédiction en cours..."):
                        predictions = model.predict([offer_text])
                        st.write("Prediction probabilities:", model.predict_proba([offer_text]))
                        st.write("Model classes:", model.named_steps['randomforestclassifier'].classes_)
                        st.write("Raw predictions:", predictions)  # Debug output
                except Exception as pred_error:
                    st.error(f"Erreur de prédiction: {str(pred_error)}")
                    st.stop()
                
                # Display results
                st.subheader("🎯 Résultats")
                
                try:
                    if predictions is not None:
                        # Convert predictions to list if needed
                        if isinstance(predictions, np.ndarray):
                            predictions = predictions.tolist()
                        elif not isinstance(predictions, list):
                            predictions = [predictions]
                        
                        # Ensure we have valid profile IDs
                        valid_predictions = [p for p in predictions if p in profils_df.index]
                        
                        if valid_predictions:
                            display_predictions(valid_predictions, profils_df, offer_text)
                        else:
                            st.warning("Aucun profil valide trouvé dans les prédictions")
                    else:
                        st.warning("Aucune prédiction générée")
                
                except Exception as display_error:
                    st.error(f"Erreur d'affichage: {str(display_error)}")
                
                # Technical details
                with st.expander("📊 Détails techniques"):
                    st.json({
                        "input_text": offer_text,
                        "predictions": predictions,
                        "features_activated": len(features.nonzero()[1]) if features is not None else 0,
                        "model_type": str(type(model))
                    })
            
            except Exception as e:
                st.error(f"Erreur inattendue: {str(e)}")
                st.stop()
elif choice == "Deep Learning": 
  run_deep_learning(profils_df)
  
elif choice == "Comparaison ML/DL":
    st.title("📊 Comparaison des Performances ML vs DL")
    
    # Vérifier les données disponibles
    has_ml = 'metrics' in st.session_state and st.session_state.metrics is not None
    has_dl = 'dl_metrics' in st.session_state and st.session_state.dl_metrics is not None
    
    if not has_ml and not has_dl:
        st.warning("Veuillez d'abord entraîner au moins un modèle (ML ou DL)")
        st.stop()
    
    # Préparer les données pour la comparaison
    comparison_data = []
    
    if has_ml:
        ml_metrics = st.session_state.metrics
        comparison_data.append({
            'Modèle': 'Machine Learning',
            'Type': getattr(st.session_state.get('model'), 'model_type', 'Random Forest'),
            'Précision': ml_metrics.get('precision', 0),
            'Exactitude': ml_metrics.get('accuracy', 0),
            'Score': (ml_metrics.get('precision', 0) + ml_metrics.get('accuracy', 0)) / 2
        })
    
    if has_dl:
        dl_metrics = st.session_state.dl_metrics
        comparison_data.append({
            'Modèle': 'Deep Learning',
            'Type': getattr(st.session_state.dl_model, 'model_type', 'ANN').upper(),
            'Précision': dl_metrics.get('accuracy', 0),  # DL retourne accuracy
            'Exactitude': dl_metrics.get('accuracy', 0),
            'Score': dl_metrics.get('accuracy', 0)
        })
    
    # Créer le DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # Afficher les résultats
    st.subheader("📋 Métriques Comparées")
    st.dataframe(df_comparison)
    
    # Visualisation
    st.subheader("📈 Visualisation des Performances")
    fig = px.bar(
        df_comparison.melt(id_vars=['Modèle', 'Type']), 
        x='Modèle', 
        y='value', 
        color='variable',
        barmode='group',
        labels={'value': 'Score', 'variable': 'Métrique'},
        title='Comparaison ML vs DL',
        color_discrete_map={
            'Précision': '#636EFA',
            'Exactitude': '#EF553B',
            'Score': '#00CC96'
        }
    )
    st.plotly_chart(fig)
    
    # Ajouter des conseils
    with st.expander("💡 Conseils d'interprétation"):
        st.write("""
        - **ML (Random Forest/Logistic Regression)**:
          - Rapide à entraîner
          - Idéal pour petits/moyens datasets
          - Moins bon sur données complexes
          
        - **DL (ANN/MLP/DNN)**:
          - Plus long à entraîner
          - Meilleur sur données complexes
          - Nécessite plus de données
        """)