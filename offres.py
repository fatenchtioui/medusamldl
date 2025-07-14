import pandas as pd
import re
import spacy

# Charger modèle NLP français
nlp = spacy.load("fr_core_news_sm")

# 🔎 Extraction de données depuis un texte libre (offre .txt ou description CSV)

from keywords_localisation import (
    extract_keywords,
    get_region_and_department,
    normalize_city_name
)
def extract_offre_info(text, titre=None, localisation=None, region="", department="", **kwargs):
    """
    Version simplifiée et unifiée
    """
    niveaux = ["junior", "confirmé", "senior", "débutant"]
    
    # Normalisation localisation
    localisation = normalize_city_name(localisation) if localisation else ""

    # Extraction compétences
    competences = re.findall(r'\b\w{3,}\b', text.lower())

    # Gestion expérience (accepte inter_exp ou experience)
    experience = kwargs.get('inter_exp') or kwargs.get('experience')
    annees_matches = re.findall(r"(\d+)\s*(ans|année)", text.lower())
    annees_exp = int(annees_matches[0][0]) if annees_matches else None
    niveau_exp = experience or next((n for n in niveaux if n in text.lower()), None)

    return {
        "titre": titre or "",
        "description": text,
        "competences": competences,
        "annees_exp": annees_exp,
        "niveau_exp": niveau_exp,
        "localisation": localisation,
        "region": region,
        "department": department,
        "inter_exp": niveau_exp
    }
# 📄 Traitement d’un fichier CSV contenant plusieurs offres
def extract_from_csv(file, index=0, region="", department=""):
    df = pd.read_csv(file)
    row = df.iloc[index]
    titre = row.get("titre", "")
    description = row.get("descriptif", "")
    localisation = row.get("lieu de prestation", "")

    full_text = f"{titre} {description}"
    return extract_offre_info(full_text, titre, localisation, region, department)

def extract_offres_from_text(text_content):
    offres = []
    current_offre = {"titre": "", "description": "", "localisation": ""}
    
    for line in text_content.split('\n'):
        line = line.strip()
        if line.startswith("Offre n°"):
            if current_offre["titre"]:
                offres.append(current_offre)
            current_offre = {"titre": line.split(":")[0], "description": "", "localisation": ""}
            
            
            if "à " in line:
                    loc = line.split("à ")[1].split(":")[0].strip()
                    current_offre["localisation"] = loc

        elif line == "---":
            continue
        elif line:
            current_offre["description"] += line + "\n"
    
    if current_offre["titre"]:
        offres.append(current_offre)
    
    return offres
from keywords_localisation import (
    extract_keywords,
    get_region_and_department,
    normalize_city_name
)

def extract_offre_info(text, titre=None, localisation=None, region="", department="", engine=None, profile_keywords=None, nlp=None):
    """
    Extrait les informations principales d'une offre à partir d'un texte brut
    """
    niveaux = ["junior", "confirmé", "senior", "débutant"]
    
    # -- Étape 1 : Normalisation de la localisation --
    localisation = normalize_city_name(localisation) if localisation else ""

    # Compléter automatiquement la région/département si engine est fourni
    # if engine and localisation:
    #     auto_region, auto_dep = get_region_and_department(localisation, engine)
    #     if not region:
    #         region = auto_region or ""
    #     if not department:
    #         department = auto_dep or ""

    # -- Étape 2 : Extraire les mots-clés (compétences) selon base de données --
    if profile_keywords:
        top_keywords = extract_keywords(text, profile_keywords)
        competences = list(top_keywords.keys())
    else:
        competences = re.findall(r'\b\w{3,}\b', text.lower())

    # -- Étape 3 : Extraction expérience --
    annees_exp = re.findall(r"(\d+)\s*(ans|année)", text.lower())
    annees_exp = int(annees_exp[0][0]) if annees_exp else None

    niveau_exp = next((n for n in niveaux if n in text.lower()), None)

    # -- Étape 4 : Lemmatisation --
    if nlp:
        tokens = [token.lemma_ for token in nlp(text.lower()) if not token.is_stop and token.is_alpha]
    else:
        tokens = []

    # -- Retour --
    return {
        "titre": titre or "",
        "description": text,
        "competences": competences,
        "competences_lemmatisees": tokens,
        "annees_exp": annees_exp,
        "niveau_exp": niveau_exp,
        "localisation": localisation,
        "region": region,
        "department": department
    }