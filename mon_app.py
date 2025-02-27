import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
import os

# Titre et description de l'application
st.set_page_config(
    page_title="Détection Précoce des Retards de Développement",
    page_icon="👶",
    layout="wide"
)

# Fonction pour charger le modèle et les préprocesseurs
@st.cache_resource
def load_resources():
    # Si le modèle n'existe pas encore, nous pouvons le créer ici
    if not os.path.exists('model.joblib'):
        st.warning("Le modèle n'a pas été trouvé. Veuillez d'abord l'entraîner et le sauvegarder.")
        return None, None, None
    
    model = joblib.load('model.joblib')
    
    # Chargement des préprocesseurs si disponibles
    scaler = None
    label_encoders = {}
    
    if os.path.exists('scaler.joblib'):
        scaler = joblib.load('scaler.joblib')
    
    if os.path.exists('label_encoders.joblib'):
        label_encoders = joblib.load('label_encoders.joblib')
    
    return model, scaler, label_encoders

# Charger les ressources
try:
    model, scaler, label_encoders = load_resources()
    models_loaded = model is not None
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles: {e}")
    models_loaded = False

# Fonction pour prétraiter les données d'entrée
def preprocess_input(input_data, scaler, label_encoders):
    """
    Prétraite les données d'entrée de la même manière que lors de l'entraînement.
    
    Args:
        input_data: DataFrame avec les données saisies par l'utilisateur
        scaler: Le scaler utilisé pour normaliser les données numériques
        label_encoders: Dictionnaire des encodeurs pour les variables catégorielles
        
    Returns:
        DataFrame prétraité prêt pour la prédiction
    """
    # Créer un DataFrame vide avec toutes les colonnes attendues par le modèle
    expected_columns = ['sexe', 'niveau_education_mere', 'revenu_mensuel_famille', 'poids_kg', 
                        'taille_cm', 'indice_IMC', 'age_marche_mois', 'age_parole_mois', 
                        'age_sassoir_mois', 'score_cognitif', 'score_moteur']
    
    processed_data = pd.DataFrame(columns=expected_columns)
    
    # Mapping entre les champs du formulaire et les colonnes attendues
    processed_data['sexe'] = [input_data['sexe'][0]]  # Supposons que vous avez ajouté cette entrée
    processed_data['niveau_education_mere'] = [input_data['niveau_education_mere'][0]]  # Idem
    processed_data['revenu_mensuel_famille'] = [input_data['revenu_mensuel_famille'][0]]
    processed_data['poids_kg'] = [input_data['weight'][0]]
    processed_data['taille_cm'] = [input_data['height'][0]]
    processed_data['indice_IMC'] = [input_data['weight'][0] / ((input_data['height'][0]/100) ** 2)]
    processed_data['age_marche_mois'] = [input_data.get('age_marche_mois', [12])[0]]
    processed_data['age_parole_mois'] = [input_data.get('age_parole_mois', [12])[0]]
    processed_data['age_sassoir_mois'] = [input_data.get('age_sassoir_mois', [6])[0]]
    processed_data['score_cognitif'] = [input_data['language_skills'][0]]
    processed_data['score_moteur'] = [input_data['motor_skills'][0]]
    
    # Appliquer le LabelEncoder aux variables catégorielles
    if label_encoders:
        for col, le in label_encoders.items():
            if col in processed_data.columns:
                processed_data[col] = le.transform(processed_data[col])
    
    # Appliquer le StandardScaler aux variables numériques
    if scaler:
        numerical_cols = ["revenu_mensuel_famille", "poids_kg", "taille_cm", "indice_IMC", 
                           "age_marche_mois", "age_parole_mois", "age_sassoir_mois", 
                           "score_cognitif", "score_moteur"]
        processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
    
    return processed_data

# Fonction pour prédire les retards de développement
def predict_delay(model, input_data, scaler, label_encoders):
    """
    Fonction pour prédire les retards de développement.
    
    Args:
        model: Le modèle chargé pour faire des prédictions
        input_data: DataFrame contenant les données d'entrée
        scaler: Le scaler utilisé pour normaliser les données
        label_encoders: Les encodeurs pour les variables catégorielles
        
    Returns:
        Tuple contenant (classes prédites, probabilités)
    """
    # Prétraiter les données
    processed_data = preprocess_input(input_data, scaler, label_encoders)
    
    # Assurer que les colonnes correspondent à celles utilisées lors de l'entraînement
    processed_data = processed_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Faire la prédiction
    prediction_classes = model.predict(processed_data)
    prediction_probabilities = model.predict_proba(processed_data)
    
    return prediction_classes, prediction_probabilities


def generate_recommendations(age_mois, prediction, scores):
    """
    Génère des recommandations personnalisées basées sur l'âge, la prédiction et les scores.
    
    Args:
        age_mois: Âge de l'enfant en mois
        prediction: La prédiction du modèle (0=pas de retard, 1=retard)
        scores: Dictionnaire contenant les scores pour différentes compétences
        
    Returns:
        Dictionnaire de recommandations par catégorie
    """
    recommendations = {
        'général': [],
        'motricité': [],
        'langage': [],
        'social': [],
        'nutrition': [],
        'environnement': []
    }
    
    # Recommandations générales basées sur le niveau de risque
    if prediction == 0:
        recommendations['général'].append("Continuez à suivre le développement normal de votre enfant.")
        recommendations['général'].append("Consultez votre pédiatre pour les visites de routine recommandées.")
    else:  # prediction == 1
        recommendations['général'].append("Une évaluation professionnelle est recommandée pour déterminer si une intervention précoce est nécessaire.")
        recommendations['général'].append("Contactez votre pédiatre pour discuter des options d'intervention disponibles.")
    
    # Recommandations générales basées sur le niveau de risque
    if prediction == 0:
        recommendations['général'].append("Continuez à suivre le développement normal de votre enfant.")
        recommendations['général'].append("Consultez votre pédiatre pour les visites de routine recommandées.")
    else:  # prediction == 1
        recommendations['général'].append("Une évaluation professionnelle est recommandée pour déterminer si une intervention précoce est nécessaire.")
        recommendations['général'].append("Contactez votre pédiatre pour discuter des options d'intervention disponibles.")
    
    # Recommandations spécifiques basées sur les scores
    # Motricité
    if scores['score_moteur'] < 0.4:
        recommendations['motricité'].append("Encouragez les activités physiques adaptées à l'âge de votre enfant.")
        recommendations['motricité'].append("Créez un espace sûr pour explorer et pratiquer de nouvelles compétences motrices.")
        if age_mois < 12:
            recommendations['motricité'].append("Pratiquez le temps sur le ventre pour renforcer les muscles du cou et du dos.")
        elif age_mois < 24:
            recommendations['motricité'].append("Encouragez la marche et l'exploration avec un soutien approprié.")
        else:
            recommendations['motricité'].append("Intégrez des jeux qui développent l'équilibre et la coordination.")
    
    # Langage
    if scores['score_cognitif'] < 0.4:
        recommendations['langage'].append("Parlez régulièrement à votre enfant en utilisant un langage clair et simple.")
        recommendations['langage'].append("Lisez des livres adaptés à son âge quotidiennement.")
        if age_mois < 12:
            recommendations['langage'].append("Répondez aux babillages et aux vocalisations de votre enfant.")
        elif age_mois < 24:
            recommendations['langage'].append("Nommez les objets et les actions dans son environnement quotidien.")
        else:
            recommendations['langage'].append("Posez des questions ouvertes et donnez-lui le temps de répondre.")
    
    # Social
    if scores.get('social_skills', 0) < 0.4:
        recommendations['social'].append("Créez des opportunités d'interaction avec d'autres enfants.")
        recommendations['social'].append("Jouez à des jeux interactifs appropriés à son âge.")
        if age_mois > 18:
            recommendations['social'].append("Encouragez les jeux de rôle et le partage.")
    
    # Nutrition
    if 'nutrition_score' in scores and scores['nutrition_score'] < 0.6:
        recommendations['nutrition'].append("Assurez-vous d'offrir une alimentation variée et équilibrée.")
        recommendations['nutrition'].append("Consultez un pédiatre ou un nutritionniste pour des conseils alimentaires adaptés.")
        if age_mois < 12:
            recommendations['nutrition'].append("Suivez les recommandations concernant l'introduction des aliments solides.")
        else:
            recommendations['nutrition'].append("Limitez les aliments transformés et riches en sucre.")
    
    # Environnement
    if 'environment_quality' in scores and scores['environment_quality'] < 0.6:
        recommendations['environnement'].append("Créez un environnement stimulant avec des jouets adaptés à son âge.")
        recommendations['environnement'].append("Établissez des routines régulières pour le sommeil, les repas et le jeu.")
        recommendations['environnement'].append("Réduisez l'exposition aux écrans selon les recommandations pour son âge.")
    
    return recommendations

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Évaluation", "Suivi", "À propos"])

# Page d'accueil
if page == "Accueil":
    st.title("Système de Détection Précoce des Retards de Développement")
    
    # Utiliser une image statique pour l'exemple ou un placeholder
    try:
        st.image("baby_image.jpg", width=600)
    except:
        st.image("mere_enfant.jpg", width=600)
    
    st.markdown("""
    ## Bienvenue dans notre application de détection précoce des retards de développement
    
    Cette application utilise l'intelligence artificielle pour aider les parents et les professionnels de santé à :
    
    * Détecter précocement les signes potentiels de retard de développement
    * Suivre la progression du développement de l'enfant
    * Recevoir des recommandations personnalisées
    
    ### Comment utiliser cette application ?
    
    1. Allez à la page **Évaluation** pour évaluer le développement actuel de votre enfant
    2. Utilisez la page **Suivi** pour enregistrer et suivre les progrès dans le temps
    
    > **Note importante** : Cette application est un outil d'aide et ne remplace pas l'avis d'un professionnel de santé.
    """)
    
    st.info("Pour commencer, sélectionnez 'Évaluation' dans le menu de gauche.")

# Page d'évaluation
elif page == "Évaluation":
    st.title("Évaluation du Développement")
    
    st.write("Veuillez remplir les informations suivantes pour évaluer le développement de votre enfant.")
    
    # Formulaire d'évaluation
    with st.form("evaluation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom de l'enfant (optionnel)")
            sexe = st.selectbox("Sexe", [("M", 0), ("F", 1)], format_func=lambda x: x[0])[0]
            age_months = st.number_input("Âge en mois", min_value=1, max_value=60, value=12)
            weight = st.number_input("Poids (kg)", min_value=0.5, max_value=30.0, value=9.0, step=0.1)
            height = st.number_input("Taille (cm)", min_value=30.0, max_value=120.0, value=75.0, step=0.5)
            age_marche_mois = st.number_input("Âge de la marche (mois)", min_value=0, max_value=36, value=12)
            age_parole_mois = st.number_input("Âge des premiers mots (mois)", min_value=0, max_value=36, value=12)
            age_sassoir_mois = st.number_input("Âge pour s'asseoir (mois)", min_value=0, max_value=36, value=6)
        
        with col2:
            niveau_education_mere = st.selectbox("Niveau d'éducation de la mère", 
                                              [("Primaire", 0), ("Secondaire", 1), ("Supérieur", 2)], 
                                              format_func=lambda x: x[0])[0]
            revenu_mensuel_famille = st.number_input("Revenu mensuel de la famille", min_value=0, value=2000)
            motor_skills = st.slider("Score moteur (0-1)", 0.0, 1.0, 0.5, 0.01, 
                                   help="Évaluez la capacité de l'enfant à se déplacer, s'asseoir, saisir des objets, etc.")
            language_skills = st.slider("Score cognitif (0-1)", 0.0, 1.0, 0.5, 0.01,
                                      help="Évaluez la capacité de l'enfant à comprendre, à s'exprimer, à utiliser des mots, etc.")
            social_skills = st.slider("Compétences sociales (0-1)", 0.0, 1.0, 0.5, 0.01,
                                    help="Évaluez la capacité de l'enfant à interagir, à répondre aux sourires, à jouer avec d'autres, etc.")
                                    
        
        st.write("Informations sur l'environnement et la nutrition:")
        col3, col4 = st.columns(2)
        
        with col3:
            nutrition_score = st.slider("Score de nutrition (0-1)", 0.0, 1.0, 0.7, 0.01,
                                      help="Évaluez la qualité et la variété de l'alimentation de l'enfant")
        
        with col4:
            environment_quality = st.slider("Qualité de l'environnement (0-1)", 0.0, 1.0, 0.7, 0.01,
                                         help="Évaluez la qualité de l'environnement de l'enfant: stimulation, sécurité, etc.")
        
        submit_button = st.form_submit_button("Évaluer")
    
    # Lorsque le formulaire est soumis
    if submit_button and models_loaded:
        # Préparer les données pour la prédiction
        input_data = pd.DataFrame({
            'sexe': [sexe],
            'niveau_education_mere': [niveau_education_mere],
            'revenu_mensuel_famille': [revenu_mensuel_famille],
            'weight': [weight],
            'height': [height],
            'age_marche_mois': [age_marche_mois],
            'age_parole_mois': [age_parole_mois],
            'age_sassoir_mois': [age_sassoir_mois],
            'motor_skills': [motor_skills],
            'language_skills': [language_skills]
        })
        
        # Faire la prédiction
        prediction_class, prediction_prob = predict_delay(model, input_data, scaler, label_encoders)
        prediction = prediction_class[0]
        probabilities = prediction_prob[0]
        
        # Créer un dictionnaire avec les scores pour les recommandations
        scores = {
            'score_moteur': motor_skills,
            'score_cognitif': language_skills,
            'social_skills': social_skills,
            'nutrition_score': nutrition_score,
            'environment_quality': environment_quality
        }
        
        # Générer des recommandations
        recommendations = generate_recommendations(age_months, prediction, scores)
        
        # Afficher les résultats
        st.header("Résultats de l'évaluation")
        
        # Afficher le niveau de risque
        risk_levels = ["Pas de retard détecté", "Retard possible"]
        risk_colors = ["green", "red"]
        
        st.markdown(f"### Niveau de risque : <span style='color:{risk_colors[prediction]}'>{risk_levels[prediction]}</span>", unsafe_allow_html=True)
        
        # Afficher les probabilités
        st.subheader("Détails de l'évaluation")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(risk_levels, probabilities * 100, color=['green', 'red'])
        ax.set_ylabel('Probabilité (%)')
        ax.set_title('Probabilité pour chaque niveau de risque')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Afficher les recommandations
        st.header("Recommandations personnalisées")
        
        for category, recs in recommendations.items():
            if recs:
                if category == 'général':
                    expander = st.expander("📋 Recommandations générales", expanded=True)
                elif category == 'motricité':
                    expander = st.expander("🏃 Motricité", expanded=True)
                elif category == 'langage':
                    expander = st.expander("🗣️ Langage", expanded=True)
                elif category == 'social':
                    expander = st.expander("👥 Compétences sociales", expanded=True)
                elif category == 'nutrition':
                    expander = st.expander("🍎 Nutrition", expanded=True)
                elif category == 'environnement':
                    expander = st.expander("🏠 Environnement", expanded=True)
                
                with expander:
                    for rec in recs:
                        st.write(f"- {rec}")
        
        # Enregistrer les résultats dans la session
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Ajouter l'évaluation actuelle à l'historique
        st.session_state.history.append({
            'date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            'name': name if name else "Enfant",
            'age_months': age_months,
            'prediction': prediction,
            'scores': scores
        })
        
        st.success("Évaluation complétée et sauvegardée dans l'historique.")

# Page de suivi
elif page == "Suivi":
    st.title("Suivi du Développement")
    
    if 'history' not in st.session_state or len(st.session_state.history) == 0:
        st.info("Aucune évaluation enregistrée. Veuillez d'abord effectuer une évaluation.")
    else:
        st.write("Historique des évaluations:")
        
        # Créer un DataFrame à partir de l'historique
        history_df = pd.DataFrame(st.session_state.history)
        
        # Afficher le tableau d'historique
        st.dataframe(history_df[['date', 'name', 'age_months', 'prediction']])
        
        # Si plusieurs évaluations sont disponibles, créer un graphique d'évolution
        if len(st.session_state.history) > 1:
            st.subheader("Évolution dans le temps")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for score_name in ['score_moteur', 'score_cognitif', 'social_skills', 'nutrition_score', 'environment_quality']:
                scores = [entry['scores'].get(score_name, 0) for entry in st.session_state.history]
                dates = [entry['date'] for entry in st.session_state.history]
                ax.plot(dates, scores, marker='o', linestyle='-', label=score_name)
            
            ax.set_ylabel('Score (0-1)')
            ax.set_title('Évolution des scores au fil du temps')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)

# Page À propos
elif page == "À propos":
    st.title("À propos de cette application")
    
    st.markdown("""
    ## Système de Détection Précoce des Retards de Développement
    
    Cette application a été conçue pour aider à la détection précoce des retards de développement chez les enfants de 0 à 5 ans.
    
    ### Comment ça marche ?
    
    L'application utilise un modèle d'apprentissage automatique entraîné sur des données de développement infantile pour évaluer le risque de retard de développement. Le modèle analyse plusieurs facteurs, notamment:
    
    * Les compétences motrices
    * Les compétences cognitives et langagières
    * L'environnement social
    * La nutrition et la croissance physique
    
    ### Avertissement
    
    Cette application est un outil d'aide à la décision et ne remplace en aucun cas l'avis d'un professionnel de santé. Si vous avez des inquiétudes concernant le développement de votre enfant, veuillez consulter un pédiatre ou un spécialiste du développement infantile.
    """)
    
    st.info("Développé dans le cadre d'un projet éducatif.")