import psycopg2
import streamlit as st
def init_db_connection():
    st.sidebar.title("🔐 Connexion Base de Données")
    db_params = {
        'host': st.sidebar.text_input("Hôte", value="localhost"),
        'database': st.sidebar.text_input("Nom de la base", value="medusa"),
        'user': st.sidebar.text_input("Utilisateur", value="postgres"),
        'password': st.sidebar.text_input("Mot de passe", type="password", value="admin"),
        'port': st.sidebar.text_input("Port", value="5432")
    }
    
    if st.sidebar.button("🔌 Se connecter"):
        try:
            # Force UTF-8 encoding for the connection
            conn = psycopg2.connect(
                **db_params,
                client_encoding='utf-8'
            )
            conn.close()
            st.session_state['db_params'] = db_params
            st.session_state['connected'] = True
            st.sidebar.success("✅ Connexion réussie!")
        except Exception as e:
            st.sidebar.error(f"❌ Échec connexion: {str(e)}")

    return st.session_state.get('connected', False)
   
