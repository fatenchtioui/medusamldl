import re
import unicodedata
import logging
from collections import Counter
from sqlalchemy import create_engine, text
from nltk.corpus import stopwords

from database import init_db_connection
from nltk.corpus import stopwords
import nltk

try:
    stop_words = set(stopwords.words('french'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))
import spacy

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download
    download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# Logger
logger = logging.getLogger(__name__)

# Stopwords personnalisés
custom_stopwords = {
    "je", "à", "et", "le", "la", "les", "un", "une", "des", "en", "du", "de", "paris", "lyon",
    "avec", "pour", "dans", "sur", "poste", "ans", "recherche", "senior", "junior", "expert",
    "ayant", "cherche", "expérience", "non"
}
stop_words = set(stopwords.words('french'))
stop_words.update(custom_stopwords)
DB_HOST="localhost"
DB_USER="postgres"
DB_PASSWORD="admin"
DB_NAME="medusa"
DB_SCHEMA="medusa"

# 3) Construire l’URL de connexion SQLAlchemy
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)

# 4) Créer l’engine en précisant le schema par défaut
engine = create_engine(
    DATABASE_URL,
    connect_args={
        # Force le search_path sur le schema défini
        "options": f"-c search_path={DB_SCHEMA}",
        "client_encoding": "utf8"
    }
)
def connect_to_database():
    """
    Retourne un engine SQLAlchemy connecté au bon schema.
    """
    try:
        engine_local = create_engine(
            DATABASE_URL,
            connect_args={
                "options": f"-c search_path={DB_SCHEMA}",
                "client_encoding": "utf8"
            }
        )
        return engine_local
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données : {e}")
        return None
# 🔑 Extraction des mots-clés existants depuis la base
def get_profile_keywords_from_postgres():
    try:
        engine = connect_to_database()
        if engine is None:
            logger.error("Connexion à la base de données échouée.")
            return set()

        query = "SELECT competences FROM profiles"
        with engine.connect() as connection:
            result = connection.execute(text(query)).fetchall()

        all_keywords = []
        for row in result:
            if row[0]:
                words = [word.strip().lower() for word in row[0].split(',') if word.strip()]
                all_keywords.extend(words)

        unique_keywords = sorted(set(all_keywords))
       # logger.info(f"{len(unique_keywords)} compétences uniques extraites")
        return set(unique_keywords)

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des compétences : {e}")
        return set()
print(get_profile_keywords_from_postgres())
# 🧠 Extraction de mots-clés d'un texte libre
def extract_keywords(text, profile_keywords):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [
        word for word in words
        if word not in stop_words and len(word) > 2 and word in profile_keywords
    ]
    return Counter(filtered)

# 🧹 Normalisation du nom de ville (accents + lowercase)
def normalize_city_name(city_name):
    if not isinstance(city_name, str):
        return ""
    try:
        normalized = unicodedata.normalize('NFKD', city_name).encode('ASCII', 'ignore').decode('utf-8')
        return normalized.lower().strip()
    except:
        return city_name.lower().strip()

# 🔍 Vérification de la ville dans la vue V_cities
def is_valid_city(city, engine):
    try:
        city = normalize_city_name(city)
        query = "SELECT COUNT(*) FROM medusa.V_cities WHERE LOWER(unaccent(label)) = :city"
        with engine.connect() as conn:
            result = conn.execute(text(query), {"city": city}).scalar()
            return result > 0
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la ville : {e}")
        return False

# 🔄 Obtenir la région et département depuis la ville
def get_region_and_department(city, engine):
    try:
        city = normalize_city_name(city)
        
        # Special case for Île-de-France
        if city == 'ile de france':
            return 'ILE-DE-FRANCE', '75'
        
        # Safely handle encoding
        try:
            city = city.encode('latin1').decode('utf-8', errors='replace')
        except:
            pass
            
        query = text("""
        SELECT region_name, department_name
        FROM medusa.cities
        WHERE LOWER(unaccent(label)) = :city
        LIMIT 1;
        """)
        
        with engine.connect().execution_options(autocommit=True) as conn:
            # Ensure parameters are properly encoded
            result = conn.execute(query, {"city": city.encode('latin1').decode('utf-8', errors='replace')}).fetchone()
            if result:
                # Handle potential encoding issues in results
                region = result[0].encode('latin1').decode('utf-8', errors='replace') if result[0] else ''
                department = result[1].encode('latin1').decode('utf-8', errors='replace') if result[1] else ''
                return region, department
                
        return None, None
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la région/département : {str(e)}")
        return None, None
