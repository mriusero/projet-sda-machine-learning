import streamlit as st
import pandas as pd
from ..FailuresDetectModel import make_predictions, save_predictions

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = [
    RandomForestClassifier(),
    LogisticRegression()
]

def page_6():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)  # TITLE
    st.markdown('<div class="header">#6 Prediction_</div>', unsafe_allow_html=True)
    texte = """
    Here is the Prediction phase.
    """
    st.markdown(texte)

    df = pd.DataFrame()

    # Affichage des modèles disponibles
    model_names = [model.__class__.__name__ for model in models]
    selected_model_name = st.selectbox("Choisissez un modèle pour les prédictions", model_names)

    if st.button("Faire des prédictions"):
        # Effectuer des prédictions pour tous les modèles
        predictions = make_predictions(models, df)

        # Afficher les prédictions pour le modèle sélectionné
        if selected_model_name in predictions:
            st.subheader(f"Prédictions pour {selected_model_name}")
            st.write(pd.DataFrame(predictions[selected_model_name], columns=['Predictions']))
        else:
            st.error(f"Aucun modèle trouvé avec le nom {selected_model_name}")

        # Sauvegarder les prédictions dans un fichier
        output_path = st.text_input("Entrez le chemin de sauvegarde des prédictions", value="./")
        if st.button("Sauvegarder les prédictions"):
            save_predictions(predictions, output_path)
            st.success(f"Les prédictions ont été sauvegardées dans {output_path}")
