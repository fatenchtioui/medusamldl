import numpy as np
import pandas as pd
import streamlit as st
import re  # Ajoutez cette ligne avec les autres imports


from profile_1 import load_profiles
def prepare_training_data(matches_df):
    features = []
    labels = []
    
    for _, row in matches_df.iterrows():
        # Text features
        text_feature = f"{row['offer_title']} {row['matched_skills']}"
        
        # Numeric features (normalized)
        exp_match = min(row['offer_exp'], row['profile_exp']) / max(row['offer_exp'], row['profile_exp'], 1)
        location_match = 1 if row['offer_location'] == row['profile_location'] else 0
        
        features.append({
            'text': text_feature,
            'exp_match': exp_match,
            'location_match': location_match
        })
        labels.append(row['label'])
    
    return features, labels

# def display_predictions(predicted_ids, profiles_df, offer_text=None):
#     """Version unifiée avec gestion des compétences correspondantes"""
#     st.subheader("🏆 Top Profils Recommandés")
    
#     if not isinstance(predicted_ids, (list, np.ndarray)):
#         st.error("Format de prédiction invalide")
#         return
    
#     # Extraction des compétences de l'offre si disponible
#     offer_skills = set()
#     if offer_text:
#         offer_skills = set(re.findall(r'\b[\w-]+\b', offer_text.lower()))
    
#     for i, profile_id in enumerate(predicted_ids[:5]):  # Affiche les top 5
#         try:
#             profile = profiles_df.iloc[profile_id] if isinstance(profiles_df, pd.DataFrame) else profiles_df.loc[profile_id]
#             profile_skills = set(re.findall(r'\b[\w-]+\b', str(profile.get('competences', '')).lower()))
            
#             with st.container():
#                 cols = st.columns([1, 4])
#                 with cols[0]:
#                     st.markdown(f"**#{i+1}**")
#                 with cols[1]:
#                     st.markdown(f"""
#                     **Poste:** {profile.get('poste', 'Non spécifié')}  
#                     **Localisation:** {profile.get('localisation', 'Non spécifié')}  
#                     **Expérience:** {profile.get('inter_exp', 'Non spécifié')}
#                     """)
                    
#                     # Affichage dynamique des compétences
#                     if offer_skills:
#                         matched_skills = offer_skills & profile_skills
#                         if matched_skills:
#                             st.markdown("**Compétences correspondantes:**")
#                             cols_skills = st.columns(4)
#                             for j, skill in enumerate(matched_skills):
#                                 cols_skills[j%4].success(f"✓ {skill}")
                    
#                     st.markdown(f"**Compétences clés:** {', '.join(list(profile_skills)[:5])}...")
                
#                 st.divider()
                
#         except Exception as e:
#             st.warning(f"Erreur affichage profil {profile_id}: {str(e)}")
# def display_predictions(predictions, profiles_df, offer_text=None):
#     """Affiche les résultats de prédiction de manière uniforme"""
#     if not isinstance(predictions, (list, np.ndarray)):
#         predictions = [predictions]  # Convertit les single values en liste
    
#     st.subheader("🏆 Top Profils Recommandés")
    
#     for i, pred in enumerate(predictions[:5]):  # Limite à 5 résultats
#         try:
#             # Gère à la fois les indices et les IDs directs
#             profile = profiles_df.iloc[pred] if isinstance(pred, (int, np.integer)) else profiles_df.loc[pred]
            
#             with st.container():
#                 cols = st.columns([1, 4])
#                 with cols[0]:
#                     st.markdown(f"**#{i+1}**")
#                 with cols[1]:
#                     st.markdown(f"""
#                     **Poste:** {profile.get('poste', 'Non spécifié')}  
#                     **Localisation:** {profile.get('localisation', 'Non spécifié')}  
#                     **Expérience:** {profile.get('inter_exp', 'Non spécifié')}
#                     """)
                    
#                     # Affichage des compétences
#                     competences = profile.get('competences', '').split(',')[:5]
#                     st.markdown(f"**Compétences clés:** {', '.join(competences)}...")
                
#                 st.divider()
                
#         except Exception as e:
#             st.warning(f"Erreur affichage profil {pred}: {str(e)}")            
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
def display_predictions(predicted_ids, profiles_df, offer_text=None):
    if offer_text:  # If we have offer text, use similarity
        vectorizer = TfidfVectorizer()
        # Vectorize all profiles
        profile_texts = profiles_df['poste'] + " " + profiles_df['competences']
        vectors = vectorizer.fit_transform(profile_texts)
        # Vectorize offer
        offer_vec = vectorizer.transform([offer_text])
        # Calculate similarities
        similarities = cosine_similarity(offer_vec, vectors).flatten()
        # Get top 5 most similar
        top_indices = similarities.argsort()[-5:][::-1]
        predicted_ids = top_indices
    
    # Rest of your display code...
    """Affiche les résultats avec highlight des compétences techniques"""
    if predicted_ids is None:
        st.error("No predictions provided")
        return
        
    if not isinstance(predicted_ids, (list, np.ndarray)):
        predicted_ids = [predicted_ids]
    
    st.subheader("🏆 Top Profils Recommandés")
    
    # Compétences techniques à vérifier
    tech_skills = profiles_df['key_word_ia'].dropna().unique().tolist()
    
    for i, profile_id in enumerate(predicted_ids[:5]):  # Top 5 résultats
        try:
            profile = profiles_df.iloc[profile_id]
            competences = set(str(profile.get('competences', '')).lower().split(','))
            
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown(f"**#{i+1}**")
                with cols[1]:
                    # Header
                    st.markdown(f"""
                    **Poste:** {profile.get('poste', 'Non spécifié')}  
                    **Localisation:** {profile.get('localisation', 'Non spécifié')}  
                    **Expérience:** {profile.get('inter_exp', 'Non spécifié')}
                    """)
                    
                    # Compétences techniques trouvées
                    found_tech = [s.strip() for s in competences if s.strip() in tech_skills ]
                    
                    #if found_tech:
                       # st.success("**Compétences techniques:** " + ", ".join(found_tech))
                    #else:
                       # st.warning("Aucune compétence technique clé trouvée")
                    
                    # Toutes les compétences
                    with st.expander("Voir toutes les compétences"):
                        st.write(", ".join([s.strip() for s in competences][:15]))  # Limite à 15
                
                st.divider()
                
        except Exception as e:
            st.warning(f"Erreur affichage profil {profile_id}: {str(e)}")