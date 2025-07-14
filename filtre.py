import re
import logging
from typing import Dict
from sentence_transformers import SentenceTransformer

# Configuration du logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Chargement du modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mapping régions
patteren_region = {
    'île-de-france': {'idf', 'ile de france', 'île-de-france', 'ile-de-france', 'iles de france'},
}

# Fonction principale d'extraction
def extract_filters(description: str, villes, regions, departments) -> Dict:
    desc = description.lower()
    filters = {
        'localisation': [],
        'region': [],
        'department': [],
        'inter_exp': None,
        'experience_level': None,
        'distance': None
    }

    # Distance
    distance_pattern = re.compile(r'(\d+)\s*(km|kms?|kilomètres?)', re.IGNORECASE)
    match = distance_pattern.search(desc)
    if match:
        filters['distance'] = int(match.group(1))
        print("Distance détectée:", filters['distance'])
        for city in villes:
            if city.lower() in desc:
                filters['localisation'].append(city)
                print(f"Ville détectée : {city}")
                break

    # N-grams pour détection géographique
    tokens = re.findall(r"\b[\w'àâäôéèëêïîçùûüÿæœ-]+\b", desc)
    ngrams = set(' '.join(tokens[i:i+n]) for n in range(1, 5) for i in range(len(tokens) - n + 1))

    for ngram in ngrams:
        if ngram in regions:
            filters['region'].append(ngram)
        elif any(ngram in patterns for patterns in patteren_region.values()):
            for region_name, patterns in patteren_region.items():
                if ngram in patterns:
                    filters['region'].append(region_name)
                    break

    if not filters['region']:
        filters['department'] = [ngram for ngram in ngrams if ngram in departments and ngram.lower() != "paris"]

    if not filters['region'] and not filters['department']:
        filters['localisation'] = [ngram for ngram in ngrams if ngram in villes]

    # Expérience - nombre seul
    single_year_pattern = re.compile(r'\b(\d+)\s*(?:ans?|années?)?\b')
    if single_match := single_year_pattern.search(desc):
        years = int(single_match.group(1))
        if years <= 2:
            filters['inter_exp'] = "0-2"
        elif 3 <= years <= 5:
            filters['inter_exp'] = "2-5"
        elif 6 <= years <= 10:
            filters['inter_exp'] = "5-10"
        else:
            filters['inter_exp'] = "10+"

    # Intervalle explicite
    interval_pattern = re.compile(r'\b(\d+)-(\d+)\s*ans?\b|\b(\d+)\+\s*ans?\b')
    for match in interval_pattern.finditer(desc):
        if match.group(3):
            filters['inter_exp'] = "10+"
        else:
            min_exp = int(match.group(1))
            max_exp = int(match.group(2))
            if min_exp == 0 and max_exp == 2:
                filters['inter_exp'] = "0-2"
            elif min_exp == 2 and max_exp == 5:
                filters['inter_exp'] = "2-5"
            elif min_exp == 5 and max_exp == 10:
                filters['inter_exp'] = "5-10"
            elif min_exp >= 10 or max_exp >= 10:
                filters['inter_exp'] = "10+"

    # Niveau d'expérience
    for level in ['junior', 'senior', 'expert', 'débutant']:
        if level in desc:
            filters['experience_level'] = level
            break

    filters['region'] = list(set(filters['region']))
    filters['department'] = list(set(filters['department']))
    filters['localisation'] = list(set(filters['localisation']))

    logger.info(f"Filtres extraits : {filters}")
    return filters
