# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pickle
# import io

# from matching_dl import MatchingDLTrainer

# def run_deep_learning(profils_df):
#     st.title("🤖 Matching Intelligent avec Deep Learning")

#     if profils_df.empty:
#         st.error("Aucun profil chargé")
#         st.stop()

#     profils_df['competence_clean'] = profils_df['competences'].str.lower().str.replace('[^\w\s]', '', regex=True)
#     all_skills = profils_df['competence_clean'].tolist()

#     vectorizer = TfidfVectorizer(max_features=500)
#     X_train = vectorizer.fit_transform(all_skills).toarray()

#     # Section Entraînement
#     if st.button("🔄 Entraîner le modèle DL"):
#         y_train = np.array([
#             1 if any(k in skills.lower() for k in profils_df['key_word_ia']) else 0
#             for skills in profils_df['competences']
#         ])

#         trainer = MatchingDLTrainer(input_dim=X_train.shape[1])
#         metrics = trainer.train(X_train, y_train, epochs=20)
#         st.session_state.dl_model = trainer
#         st.session_state.dl_vectorizer = vectorizer
#         st.success("✅ Modèle entraîné avec succès !")
#         st.json(metrics)

#     # Section Export/Import du modèle
#     # Section Export/Import du modèle
#     st.subheader("📦 Gestion du modèle")
#     col1, col2 = st.columns(2)

#     with col1:
#         if 'dl_model' in st.session_state and 'dl_vectorizer' in st.session_state:
#             model_bytes = st.session_state.dl_model.export_model(st.session_state.dl_vectorizer)
#             st.download_button(
#                 label="💾 Exporter le modèle",
#                 data=model_bytes,
#                 file_name="matching_model_and_vectorizer.pkl",
#                 mime="application/octet-stream"
#             )
#         else:
#             st.warning("Entraînez un modèle avant l'export")

#     with col2:
#         uploaded_file = st.file_uploader("📤 Importer un modèle", type=["pkl"])
#         if uploaded_file is not None:
#             try:
#                 bytes_data = uploaded_file.read()
#                 trainer = MatchingDLTrainer(input_dim=500)  # Doit correspondre à max_features du TF-IDF
#                 loaded_vectorizer = trainer.load_model(bytes_data)  # Charge modèle ET vectorizer
#                 st.session_state.dl_model = trainer
#                 st.session_state.dl_vectorizer = loaded_vectorizer
#                 st.success("Modèle et vectorizer chargés avec succès !")
#             except Exception as e:
#                 st.error(f"Erreur lors du chargement : {str(e)}")

#     # Section Prédiction (utilise TOUJOURS st.session_state.dl_model)
#     if 'dl_model' in st.session_state and 'dl_vectorizer' in st.session_state:
#         # ... (le reste du code de prédiction reste inchangé)
#         st.subheader("🔮 Prédire des correspondances")

#         if "offer_text" not in st.session_state:
#             st.session_state.offer_text = "développeur fullstack java"

#         input_offer = st.text_area(
#             "Texte de l'offre",
#             value=st.session_state.offer_text,
#             height=100
#         )
#         st.session_state.offer_text = input_offer

#         if st.button("🔍 Prédire les profils correspondants"):
#             offer_vec = st.session_state.dl_vectorizer.transform([input_offer.lower()]).toarray()
#             similarities = cosine_similarity(offer_vec, X_train).flatten()
#             top_indices = np.argsort(similarities)[-5:][::-1]

#             st.success("🏆 Top 5 Profils Recommandés")
#             for i, idx in enumerate(top_indices):
#                 row = profils_df.iloc[idx]
#                 with st.container():
#                     cols = st.columns([1, 4])
#                     with cols[0]:
#                         st.metric(f"#{i+1}", f"{similarities[idx]:.1%}")
#                     with cols[1]:
#                         st.markdown(f"""
#                         **Poste:** {row['poste']}  
#                         **Localisation:** {row['localisation']}  
#                         **Expérience:** {row['inter_exp']}
#                         """)
#                         with st.expander("Compétences"):
#                             st.write(row['competences'])
#                     st.divider()
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle

from matching_dl import MatchingDLTrainer

def run_deep_learning(profils_df):
    st.title("🤖 Matching Intelligent avec Deep Learning")

    if profils_df.empty:
        st.error("Aucun profil chargé")
        st.stop()

    # Configuration du modèle dans la sidebar
    st.sidebar.header("Configuration du Modèle")
    model_type = st.sidebar.selectbox(
        "Type de modèle",
        options=["ann", "mlp", "dnn"],
        index=0,
        format_func=lambda x: {
            "ann": "ANN - Réseau Standard",
            "mlp": "MLP - Multicouche",
            "dnn": "DNN - Profond (3+ couches)"
        }[x]
    )
    
    epochs = st.sidebar.slider("Nombre d'epochs", 5, 100, 20)
    lr = st.sidebar.selectbox("Taux d'apprentissage", 
                             [1e-2, 1e-3, 1e-4], 
                             index=1)

    profils_df['competence_clean'] = profils_df['competences'].str.lower().str.replace('[^\w\s]', '', regex=True)
    all_skills = profils_df['competence_clean'].tolist()

    vectorizer = TfidfVectorizer(max_features=500)
    X_train = vectorizer.fit_transform(all_skills).toarray()

    # Section Entraînement
    if st.button("🔄 Entraîner le modèle DL"):
        y_train = np.array([
            1 if any(k in skills.lower() for k in profils_df['key_word_ia']) else 0
            for skills in profils_df['competences']
        ])

        trainer = MatchingDLTrainer(input_dim=X_train.shape[1], model_type=model_type)
        metrics = trainer.train(X_train, y_train, epochs=epochs, lr=lr)
        st.session_state.dl_model = trainer
        st.session_state.dl_vectorizer = vectorizer
        st.success("✅ Modèle entraîné avec succès !")
        st.json(metrics)

        # Afficher des informations sur le modèle
        st.subheader("📊 Architecture du modèle")
        if model_type == "ann":
            st.write("**ANN (Artificial Neural Network)** - 1 couche cachée (256 neurones)")
        elif model_type == "mlp":
            st.write("**MLP (Multi-Layer Perceptron)** - 2 couches cachées (512 → 256 neurones)")
        else:
            st.write("**DNN (Deep Neural Network)** - 4 couches cachées (1024 → 512 → 256 → 128 neurones)")

    # Section Export/Import du modèle
    st.subheader("📦 Gestion du modèle")
    col1, col2 = st.columns(2)

    with col1:
        if 'dl_model' in st.session_state and 'dl_vectorizer' in st.session_state:
            model_bytes = st.session_state.dl_model.export_model(st.session_state.dl_vectorizer)
            st.download_button(
                label="💾 Exporter le modèle",
                data=model_bytes,
                file_name="matching_model_and_vectorizer.pkl",
                mime="application/octet-stream"
            )
        else:
            st.warning("Entraînez un modèle avant l'export")

    with col2:
        uploaded_file = st.file_uploader("📤 Importer un modèle", type=["pkl"])
        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.read()
                trainer = MatchingDLTrainer(input_dim=500, model_type="ann")  # Type par défaut, sera écrasé
                loaded_vectorizer = trainer.load_model(bytes_data)
                st.session_state.dl_model = trainer
                st.session_state.dl_vectorizer = loaded_vectorizer
                st.success(f"Modèle {trainer.model_type} et vectorizer chargés avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {str(e)}")

    # Section Prédiction
    if 'dl_model' in st.session_state and 'dl_vectorizer' in st.session_state:
        st.subheader("🔮 Prédire des correspondances")

        if "offer_text" not in st.session_state:
            st.session_state.offer_text = "développeur fullstack java"

        input_offer = st.text_area(
            "Texte de l'offre",
            value=st.session_state.offer_text,
            height=100
        )
        st.session_state.offer_text = input_offer

        if st.button("🔍 Prédire les profils correspondants"):
            offer_vec = st.session_state.dl_vectorizer.transform([input_offer.lower()]).toarray()
            
            # Utilisation du modèle DL pour les probabilités
            profile_probs = st.session_state.dl_model.predict_proba(X_train)
            
            # Combinaison des similarités cosinus et des probabilités du modèle
            similarities = cosine_similarity(offer_vec, X_train).flatten()
            combined_scores = 0.7 * profile_probs + 0.3 * similarities  # Pondération
            
            top_indices = np.argsort(combined_scores)[-5:][::-1]

            st.success("🏆 Top 5 Profils Recommandés")
            for i, idx in enumerate(top_indices):
                row = profils_df.iloc[idx]
                with st.container():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        st.metric(f"#{i+1}", f"{combined_scores[idx]:.1%}")
                    with cols[1]:
                        st.markdown(f"""
                        **Poste:** {row['poste']}  
                        **Localisation:** {row['localisation']}  
                        **Expérience:** {row['inter_exp']}
                        """)
                        with st.expander("Compétences"):
                            st.write(row['competences'])
                    st.divider()