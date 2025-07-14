import pandas as pd
import re
import spacy
from sqlalchemy import create_engine, text
import streamlit as st

# Chargement du modèle NLP (chargé une seule fois)
nlp = spacy.load("fr_core_news_sm")

def categorize_experience(exp):
    try:
        exp = float(exp)
        if exp < 2: return "0-2 ans"
        elif 2 <= exp < 5: return "2-5 ans"
        elif 5 <= exp < 10: return "5-10 ans"
        else: return "10+ ans"
    except:
        return "Non spécifié"

def preprocess_row(row):
    """Handle text preprocessing with encoding safety"""
    try:
        # Safely handle text fields
        poste = safe_text_processing(row['poste'])
        competences = safe_text_processing(row['competences'])
        key_word_ia = safe_text_processing(row['key_word_ia']) if row['key_word_ia'] else ""
        
        # Rest of your existing processing logic...
        
    except Exception as e:
        st.error(f"Erreur de prétraitement pour un profil : {str(e)}")
        return pd.Series()

def safe_text_processing(text):
    """Safely process text with encoding handling"""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text)
    # Remove problematic characters
    text = text.replace('\u2019', "'")  # Replace smart quotes
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text
@st.cache_data(show_spinner=False)
def load_profiles():
    """Charge et prétraite les profils de la base de données"""
    if 'db_params' not in st.session_state:
        st.error("Connexion à la base de données non configurée")
        return pd.DataFrame()

    try:
        # Create engine with UTF-8 encoding
        engine = create_engine(
            f"postgresql+psycopg2://{st.session_state['db_params']['user']}:"
            f"{st.session_state['db_params']['password']}@"
            f"{st.session_state['db_params']['host']}:"
            f"{st.session_state['db_params']['port']}/"
            f"{st.session_state['db_params']['database']}",
            connect_args={'client_encoding': 'utf-8'}
        )

        query = """
        SELECT poste, competences, key_word_ia, region, department, 
               level_experience, inter_exp, localisation
        FROM medusa.profiles
        LIMIT 10000;
        """

        # Use direct SQLAlchemy connection for better encoding handling
        with engine.connect().execution_options(autocommit=True) as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys()).fillna('')

        if df.empty:
            st.warning("Aucun profil trouvé dans la base de données.")
            return pd.DataFrame()

        # Handle encoding in preprocessing
        def safe_decode(text):
            if isinstance(text, bytes):
                return text.decode('utf-8', errors='replace')
            return str(text)

        # Apply preprocessing with encoding safety
        processed = df.apply(lambda row: preprocess_row(row.map(safe_decode)), axis=1)
        df_final = pd.concat([df, processed], axis=1)
        return df_final

    except Exception as e:
        st.error(f"Erreur lors du chargement des profils : {str(e)}")
        return pd.DataFrame()