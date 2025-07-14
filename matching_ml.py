import datetime
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import re  # Ajoutez cette ligne avec les autres imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import re
from datetime import datetime

class MatchingTrainer:
    # def __init__(self):
    #     self._create_model()
    #     self.feedback_data = []
    #     self.trained = False
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self._create_model()
        self.feedback_data = []
        self.trained = False
    
    # def _create_model(self):
    #     """Initialisation séparable pour le pickling"""
    #     self.model = make_pipeline(
    #         TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    #         RandomForestClassifier(n_estimators=200, max_depth=10)
    #     )
    #     # On recrée le pattern à chaque fois plutôt que de le sérialiser
    #     self._update_skill_pattern()
    def _create_model(self):
        """Initialise le modèle ML selon le type choisi"""
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        if self.model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=200, max_depth=10)
        elif self.model_type == "logistic_regression":
            classifier = LogisticRegression(max_iter=1000)
        elif self.model_type == "svm":
            classifier = SVC(probability=True)
        else:
            raise ValueError(f"Modèle ML non supporté : {self.model_type}")

        self.model = make_pipeline(vectorizer, classifier)
        self._update_skill_pattern()

    def _update_skill_pattern(self):
        """Recrée le pattern d'extraction des compétences"""
        self.skill_pattern = re.compile(r'\b[\w-]+\b')
    
    def __reduce__(self):
        """Méthode spéciale pour le pickling"""
        return (self.__class__, (), self.__getstate__())
    
    def __getstate__(self):
        """Contrôle ce qui est sérialisé"""
        state = self.__dict__.copy()
        # Exclure le pattern qui n'est pas sérialisable
        if 'skill_pattern' in state:
            del state['skill_pattern']
        return state
    
    def __setstate__(self, state):
        """Reconstruction après unpickling"""
        self.__dict__.update(state)
        self._create_model()  # Recrée les éléments non sérialisables
    
    # ... (le reste de vos méthodes existantes)
       # self.skill_pattern = re.compile(r'\b[\w-]+\b')  # Pour extraire les compétences
    def extract_skills(self, text):
        """Version sans stockage du pattern"""
        if not hasattr(self, '_skill_pattern'):
            self._skill_pattern = re.compile(r'\b[\w-]+\b')
        return set(self._skill_pattern.findall(text.lower()))   
    # def train(self, X_train, y_train):
    #     """Entraîne le modèle sur les données historiques"""
    #     self.model.fit(X_train, y_train)
    #     self.trained = True
    #     return self.evaluate(X_train, y_train)
    def train(self, X_train, y_train):
        """Entraîne le modèle sur les données historiques"""
        self.model.fit(X_train, y_train)
        self.trained = True
        self.metrics = self.evaluate(X_train, y_train)  # Stocke les métriques
        return self.metrics
    def evaluate(self, X, y):
        """Évalue la performance du modèle"""
        if not self.trained:
            raise Exception("Modèle non entraîné")
        
        y_pred = self.model.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            #'trained_at': str(datetime.datetime.now())
        }
    
    # Dans matching_ml.py
    def predict(self, offer_text, profiles_df=None):
        """Version optimisée avec paramètre optionnel"""
        if not self.trained:
            raise Exception("Modèle non entraîné")
        
        # Si profiles_df n'est pas fourni, utilisez juste le modèle
        if profiles_df is None:
            features = self.model.named_steps['tfidfvectorizer'].transform([offer_text])
            return self.model.named_steps['randomforestclassifier'].predict(features)
        
        # Sinon, utilisez la version avancée avec matching des compétences
        offer_skills = self.extract_skills(offer_text)
        matching_profiles = []
        
        for idx, profile in profiles_df.iterrows():
            profile_skills = self.extract_skills(profile.get('competences', ''))
            common_skills = offer_skills & profile_skills
            if common_skills:
                matching_profiles.append((idx, len(common_skills)))
        
        matching_profiles.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in matching_profiles[:5]]
    def add_feedback(self, text, is_positive):
        """Ajoute un feedback pour le ré-entraînement"""
        self.feedback_data.append((text, 1 if is_positive else 0))
    
    def is_performant(self):
        """Détermine si le modèle est assez bon pour l'export"""
        if not self.feedback_data:
            return False
            
        X_fb = [x for x, _ in self.feedback_data]
        y_fb = [y for _, y in self.feedback_data]
        metrics = self.evaluate(X_fb, y_fb)
        
        return metrics['accuracy'] > 0.85
    
    def export(self):
        """Exporte le modèle sous forme binaire"""
        return pickle.dumps({
            'model': self.model,
            'metadata': self.evaluate() if self.feedback_data else None
        })
    # Dans matching_ml.py
    def export_model(self):
        """Version finale corrigée"""
        return {
            'pipeline': self.model,  # Le pipeline sklearn complet
            'vectorizer': self.model.named_steps['tfidfvectorizer'],
            'classifier': self.model.named_steps['randomforestclassifier'],
            'metadata': {
                'performance': getattr(self, 'metrics', None),
                'training_date': datetime.now().isoformat() if hasattr(self, 'metrics') else None,
                'model_type': 'sklearn.Pipeline'
            }
        }
    # def export_model(self):
    #     """Format standard pour l'export"""
    #     return {
    #         'model': self.model,
    #         'metadata': {
    #            # 'training_date': datetime.now().isoformat(),
    #             'performance': self.evaluate(),
    #             'model_type': str(type(self.model))
    #         }
    #     }