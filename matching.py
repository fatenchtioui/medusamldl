import pandas as pd
from typing import Dict, List, Union, Set
import streamlit as st
import re

from fonction import parse_experience_strings, safe_extract
def normalize_skill(skill: str) -> str:
    """Normalisation plus permissive des compétences"""
    skill = str(skill).lower().strip()
    skill = ''.join(c for c in skill if c.isalnum() or c in {' ', '-', '_'})
    return ' '.join(skill.split())  # Supprime les espaces multiples

def clean_skills(skills: Union[str, list, None]) -> set:
    """Nettoyage robuste des compétences"""
    if not skills:
        return set()
    
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',')]
    elif isinstance(skills, list):
        skills = [str(s).strip() for s in skills]
    else:
        return set()
    
    return {normalize_skill(s) for s in skills if s}

# def calculate_match_label(offer: Dict, profile: Dict, min_skills: int = 2) -> int:
#     """Nouvelle version avec 4 niveaux de matching"""
#     try:
#         # 1. Vérification des compétences (critère obligatoire)
#         offer_skills = clean_skills(offer.get('competences', []))
#         profile_skills = clean_skills(profile.get('competences', ''))
#         common_skills = offer_skills & profile_skills
        
#         if len(common_skills) < min_skills:
#             return 0  # Pas assez de compétences communes
        
#         # 2. Vérification localisation
#         offer_loc = offer.get('localisation', [])
#         profile_loc = str(profile.get('localisation', '')).lower()
#         profile_region = str(profile.get('region', '')).lower()
#         profile_dept = str(profile.get('department', '')).lower()
        
#         loc_match = False
#         if not offer_loc:  # Si pas de localisation demandée, considéré comme match
#             loc_match = True
#         else:
#             offer_locs = {str(l).lower() for l in (offer_loc if isinstance(offer_loc, list) else [offer_loc])}
#             profile_locs = {profile_loc, profile_region, profile_dept}
#             loc_match = len(offer_locs & profile_locs) > 0
        
#         # 3. Vérification expérience
#         offer_exp = str(offer.get('experience_level', '')).lower()
#         profile_exp = str(profile.get('inter_exp', '')).lower()
        
#         exp_match = False
#         if not offer_exp or offer_exp == 'none':  # Si pas d'exigence d'expérience
#             exp_match = True
#         else:
#             exp_match = (offer_exp in profile_exp) or (profile_exp in offer_exp)
        
#         # 4. Attribution du label selon les combinaisons
#         if loc_match and exp_match:
#             return 3  # Match complet (compétences + localisation + expérience)
#         elif loc_match:
#             return 2  # Compétences + localisation
#         elif exp_match:
#             return 1  # Compétences + expérience
#         else:
#             return 1  # Seulement compétences
    
#     except Exception as e:
#         print(f"Error in matching: {str(e)}")
#         return 0

def normalize_text(text: Union[str, List]) -> str:
    """Normalise un texte pour la comparaison"""
    if isinstance(text, list):
        text = ' '.join(str(x) for x in text)
    return str(text).lower().strip()


def location_match(offer_loc: Union[str, List], profile_loc: str, 
                  profile_region: str, profile_department: str) -> bool:
    """Version plus permissive du matching géographique"""
    # Si l'offre ne précise pas de localisation -> match automatique
    if not offer_loc or (isinstance(offer_loc, list) and not any(offer_loc)):
        return True
    
    # Normalisation des localisations
    offer_locs = {normalize_text(loc) for loc in ([offer_loc] if isinstance(offer_loc, str) else offer_loc)}
    profile_locs = {
        normalize_text(profile_loc),
        normalize_text(profile_region),
        normalize_text(profile_department)
    }
    
    # Match si au moins une correspondance
    return len(offer_locs & profile_locs) > 0
import time
from typing import Dict, List, Union
import pandas as pd
import streamlit as st

def location_match(offer_loc: Union[str, List], profile: Dict) -> bool:
    """Version ultra-optimisée du matching géographique"""
    if not offer_loc or (isinstance(offer_loc, list) and not offer_loc):
        return True
    
    profile_locs = {
        str(profile.get('localisation', '')).lower(),
        str(profile.get('region', '')).lower(),
        str(profile.get('department', '')).lower()
    }
    
    if isinstance(offer_loc, str):
        return str(offer_loc).lower() in profile_locs
    return any(str(loc).lower() in profile_locs for loc in offer_loc)
def experience_match(offer_exp: str, profile: Dict) -> bool:
    """Matching d'expérience optimisé avec gestion des cas NULL"""
    profile_exp = str(profile.get('inter_exp', '')).lower()
    if not offer_exp or offer_exp == 'none':
        return True
    return (offer_exp in profile_exp) or (profile_exp in offer_exp)

def calculate_match_label(offer: Dict, profile: Dict) -> Dict:
    """Calcule le niveau de matching et retourne un dictionnaire complet"""
    result = {
        'competences_match': False,
        'localisation_match': False,
        'experience_match': False,
        'label': 0,
        'reason': "Pas assez de compétences communes"
    }
    # 1. Vérification OBLIGATOIRE des compétences
    offer_skills = clean_skills(offer.get('competences', []))
    profile_skills = clean_skills(profile.get('competences', ''))
    common_skills = offer_skills & profile_skills
    min_skills: int = 2
    if len(common_skills) < min_skills:
        return result  # label reste à 0

    # 2. Si on arrive ici, les compétences sont suffisantes
    result.update({
        'competences_match': True,
        'nb_common_skills': len(common_skills),
        'matched_skills': list(common_skills)
    })
    try:
        # 1. Vérification des compétences
        offer_skills = clean_skills(offer.get('competences', []))
        profile_skills = clean_skills(profile.get('competences', ''))
        common_skills = offer_skills & profile_skills
        result['nb_common_skills'] = len(common_skills)
        result['matched_skills'] = list(common_skills)
        
        if len(common_skills) == 0:
            return result
        
        result['competences_match'] = True

        # 2. Vérification localisation
        offer_loc = offer.get('localisation', [])
        profile_loc = str(profile.get('localisation', '')).lower()
        profile_region = str(profile.get('region', '')).lower()
        profile_dept = str(profile.get('department', '')).lower()
        
        if not offer_loc:
            result['localisation_match'] = True
            result['localisation_reason'] = "Pas d'exigence de localisation"
        else:
            offer_locs = {str(l).lower() for l in (offer_loc if isinstance(offer_loc, list) else [offer_loc])}
            profile_locs = {profile_loc, profile_region, profile_dept}
            result['localisation_match'] = len(offer_locs & profile_locs) > 0
            result['localisation_reason'] = "Match" if result['localisation_match'] else "Non-match"
        
        # 3. Vérification expérience
        offer_exp = str(offer.get('experience_level', '')).lower()
        profile_exp = str(profile.get('inter_exp', '')).lower()
        
        if not offer_exp or offer_exp == 'none':
            result['experience_match'] = True
            result['experience_reason'] = "Pas d'exigence d'expérience"
        else:
            result['experience_match'] = (offer_exp in profile_exp) or (profile_exp in offer_exp)
            result['experience_reason'] = "Match" if result['experience_match'] else "Non-match"
        
        # 4. Détermination du label final
        if result['competences_match']:
            if result['localisation_match'] and result['experience_match']:
                result['label'] = 3
                result['reason'] = "Match complet (compétences + localisation + expérience)"
            elif result['localisation_match'] or result['experience_match']:
                result['label'] = 2
                reason_parts = []
                if result['localisation_match']:
                    reason_parts.append("localisation")
                if result['experience_match']:
                    reason_parts.append("expérience")
                result['reason'] = f"Match partiel (compétences + {' + '.join(reason_parts)})"
            else:
                result['label'] = 1
                result['reason'] = "Match de compétences seulement"
        
        return result
    
    except Exception as e:
        print(f"Error in matching: {str(e)}")
        result['error'] = str(e)
        return result

def match_all_offers_to_profiles(offers: List[Dict], profiles_df: pd.DataFrame, min_skills: int = 2) -> pd.DataFrame:
    """Version mise à jour avec raison du matching"""
    results = []
    total_ops = len(offers) * len(profiles_df)
    
    # Initialisation barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    for i, offer in enumerate(offers):
        offer_skills = clean_skills(offer.get('competences', []))
        offer_loc = offer.get('localisation', [])
        offer_exp = str(offer.get('experience_level', '')).lower()
        
        for j, (_, profile) in enumerate(profiles_df.iterrows()):
            try:
                profile_skills = clean_skills(profile.get('competences', ''))
                common_skills = offer_skills & profile_skills
                
                if len(common_skills) >= min_skills:
                    loc_ok = location_match(offer_loc, profile)
                    exp_ok = experience_match(offer_exp, profile)
                    
                    # Déterminer la raison du matching
                    if loc_ok and exp_ok:
                        reason = "Match complet (compétences + localisation + expérience)"
                        label = 3
                    elif loc_ok:
                        reason = "Match localisation + compétences"
                        label = 2
                    elif exp_ok:
                        reason = "Match expérience + compétences"
                        label = 2
                    else:
                        reason = "Match compétences seulement"
                        label = 1
                    
                    results.append({
                        'offer_id': i,
                        'profile_id': profile.name,
                        'label': label,
                        'matching_reason': reason,  # Ajout de ce champ
                        'offer_title': offer.get('Offre', f"Offre {i+1}"),
                        'profile_poste': safe_extract(profile, 'poste'),
                        'matched_skills': ', '.join(common_skills),
                        'nb_common_skills': len(common_skills),
                        'location_match': loc_ok,
                        'experience_match': exp_ok,
                        'offer_location': str(offer_loc),
                        'profile_location': str(profile.get('localisation', '')),
                        'offer_exp': offer_exp,
                        'profile_exp': str(profile.get('inter_exp', ''))
                    })
            
            except Exception as e:
                print(f"Erreur offre {i} profil {j}: {str(e)}")
                continue
            # Mise à jour progression plus fréquente
            if (i * len(profiles_df) + j) % 100 == 0:  # Tous les 100 profils
                progress = min(0.99, (i * len(profiles_df) + j) / total_ops)
                progress_bar.progress(progress)
                status_text.text(f"Traitement {i+1}/{len(offers)} offres | Profil {j+1}/{len(profiles_df)}")
    
    # Finalisation
    progress_bar.progress(1.0)
    status_text.text("✅ Traitement terminé !")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)
    


def filter_and_rank_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Filtre et classe les résultats selon le nouveau système de labels"""
    if matches_df.empty:
        return matches_df
    
    # Calcul du score (vous pouvez ajuster ces poids)
    matches_df['score'] = (
        matches_df['label'] * 100 +
        matches_df['nb_common_skills'] * 10 +
        matches_df['location_match'].astype(int) * 30 +
        matches_df['experience_match'].astype(int) * 30
    )
    
    # Tri par score décroissant puis par nombre de compétences communes
    return matches_df.sort_values(['score', 'nb_common_skills'], ascending=[False, False])
def filter_and_rank_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Version robuste avec vérification des colonnes"""
    required_cols = ['label', 'nb_common_skills', 'location_match', 'experience_match']
    
    if matches_df.empty:
        return matches_df
    
    # Vérification des colonnes
    missing_cols = [col for col in required_cols if col not in matches_df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour le scoring: {missing_cols}")
    
    # Calcul du score
    matches_df['score'] = (
        matches_df['label'] * 40 +
        matches_df['nb_common_skills'] * 15 +
        matches_df['location_match'].astype(int) * 30 +
        matches_df['experience_match'].astype(int) * 25
    )
    
    # Bonus pour les matchs parfaits
    perfect_match = matches_df['location_match'] & matches_df['experience_match']
    matches_df['score'] += perfect_match.astype(int) * 50
    
    return matches_df.sort_values(['score', 'nb_common_skills'], ascending=[False, False])

def create_matching_stats_chart(df: pd.DataFrame):
    """Crée un graphique Plotly des statistiques"""
    import plotly.express as px
    
    stats = pd.DataFrame({
        'Type': ['Compétences', 'Localisation', 'Expérience'],
        'Matchs': [
            df['nb_common_skills'].mean(),
            df['location_match'].mean() * 100,
            df['experience_match'].mean() * 100
        ]
    })
    
    fig = px.bar(stats, x='Type', y='Matchs', 
                 title='Statistiques des Matching',
                 labels={'Matchs': 'Taux de match (%)'},
                 color='Type')
    fig.update_yaxes(range=[0, 100])
    return fig

def clean_skills(skills: Union[str, list, None]) -> set:
    """Nettoyage robuste des compétences"""
    if not skills:
        return set()
    
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',')]
    
    return {s.lower().strip() for s in skills if s and isinstance(s, str)}