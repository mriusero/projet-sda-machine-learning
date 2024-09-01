import streamlit as st



def page_5():
    st.markdown('<div class="header">#5 Training_</div>', unsafe_allow_html=True)
    texte = """
    Here is the Training phase.
    
    ____________________________________________________
    
    1. Mesurer les Performances de Prédiction
Les métriques couramment utilisées pour évaluer les modèles de survie et les prédictions de Remaining Useful Life (RUL) incluent :

Concordance Index (C-Index): C'est une mesure de la qualité de l'ordre des prédictions. Elle évalue la capacité du modèle à classer correctement les individus en termes de risque.
Mean Absolute Error (MAE) sur les prédictions RUL: Cela mesure la précision des prédictions de RUL en comparant la valeur prédite avec la valeur réelle.
Brier Score: Une mesure pour les prédictions de probabilités.
    
    
    
    """
    st.markdown(texte)



