import streamlit as st
def parse_experience_string(exp_value):
    exp_value = str(exp_value).strip()

    # Nettoyage : supprimer caractères non pertinents sauf chiffres, + et -
    cleaned = ''.join(c for c in exp_value if c.isdigit() or c in ['-', '+'])

    try:
        if '-' in cleaned:
            parts = cleaned.split('-')
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                min_exp = int(parts[0])
                if min_exp <= 2:
                    return "0-2"
                elif min_exp <= 5:
                    return "2-5"
                elif min_exp <= 10:
                    return "5-10"
                else:
                    return "10+"
        elif '+' in cleaned:
            number = cleaned.replace('+', '')
            if number.isdigit():
                val = int(number)
                return "10+" if val >= 10 else "5-10"
        elif cleaned.isdigit():
            val = int(cleaned)
            if val <= 2:
                return "0-2"
            elif val <= 5:
                return "2-5"
            elif val <= 10:
                return "5-10"
            else:
                return "10+"
    except Exception as e:
        st.warning(f"Erreur parsing expérience : '{exp_value}' - {e}")

    return None  # Si tout échoue
import re

def parse_experience_strings(exp_value):
    if not exp_value or not isinstance(exp_value, str):
        return None

    text = exp_value.lower().replace(',', '.').strip()
    text = re.sub(r'[–—−]', '-', text)  # Normaliser les tirets

    # Traiter d'abord les parenthèses : "Niveau 3 (8-12 ans)"
    match_parens = re.search(r'\((.*?)\)', text)
    if match_parens:
        return parse_experience_strings(match_parens.group(1))

    # Chercher une plage d'années : ex "4-5 ans", "8 - 12", "7-8 ans"
    match_range = re.search(r'(\d+)\s*-\s*(\d+)\s*(ans?|années?)?', text)
    if match_range:
        min_exp = int(match_range.group(1))
        return categorize_experience(min_exp)

    # Chercher un nombre suivi de + : "+12", "12+", etc.
    match_plus = re.search(r'\+?\s*(\d+)\s*\+?', text)
    if match_plus:
        exp = int(match_plus.group(1))
        return categorize_experience(exp)

    # Dernier recours : un nombre d'années seul
    match_any = re.search(r'(\d+)\s*(an|ans|année|années)', text)
    if match_any:
        exp = int(match_any.group(1))
        return categorize_experience(exp)

    return None
def safe_extract(data, key):
                        if key not in data:
                            return ""
                        val = data[key]
                        if isinstance(val, list):
                            return val[0] if len(val) > 0 else ""
                        return str(val) if val else ""


def categorize_experience(years):
    if years <= 2:
        return "0-2"
    elif years <= 5:
        return "2-5"
    elif years <= 10:
        return "5-10"
    else:
        return "10+"
def safe_lower(value):
    """Convertit une valeur en string et la met en minuscule de manière sécurisée"""
    if isinstance(value, list):
        value = value[0] if len(value) > 0 else ""
    return str(value).lower()
    