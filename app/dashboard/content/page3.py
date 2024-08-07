import streamlit as st

def page_3():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#3 Cleaning_</div>', unsafe_allow_html=True)    
    texte = """


"""
    st.markdown(texte)    

    cleaning = """
### 2) Cleaning_

- **Gestion des valeurs manquantes** :
  - **Suppression** : Supprimez les lignes ou colonnes avec des valeurs manquantes si elles sont trop nombreuses.
  - **Imputation** : Remplacez les valeurs manquantes par la moyenne, la médiane, ou une valeur estimée par d'autres techniques.

- **Correction des erreurs** :
  - Rectifiez les erreurs typographiques ou les incohérences dans les données (par exemple, `Paris` vs `paris`).

- **Gestion des valeurs aberrantes** :
  - **Suppression** : Retirez les points de données aberrants s’ils sont peu nombreux et ne sont pas représentatifs.
  - **Transformation** : Appliquez des transformations pour réduire l'impact des valeurs aberrantes (logarithmique, etc.).

- **Encodage des variables catégorielles** :
  - **Encodage One-Hot** : Convertissez les variables catégorielles en variables binaires.
  - **Encodage Label** : Assignez des valeurs numériques uniques aux catégories.

- **Normalisation / Standardisation** :
  - **Normalisation** : Mettez à l’échelle les variables entre 0 et 1.
  - **Standardisation** : Transformez les variables pour avoir une moyenne de 0 et un écart-type de 1.

### 3) Transformation_

- **Création de nouvelles fonctionnalités** :
  - **Ingénierie des caractéristiques** : Créez de nouvelles variables à partir des variables existantes (par exemple, combiner des dates pour créer des variables temporelles).

- **Réduction de dimensionnalité** :
  - Utilisez des techniques comme PCA (Analyse en Composantes Principales) si le jeu de données a un grand nombre de variables.

- **Détection et gestion des déséquilibres** :
  - Si les classes sont déséquilibrées, appliquez des techniques comme le sur-échantillonnage ou le sous-échantillonnage.

### 4) Validation & Préparation_

- **Séparation des données** :
  - Divisez le jeu de données en ensembles d’entraînement, de validation et de test (souvent dans les proportions 70/15/15 ou 80/10/10).

- **Validation croisée** :
  - Appliquez une validation croisée pour évaluer la robustesse de votre modèle sur différentes sous-parties des données.

- **Sauvegarde des données** :
  - Enregistrez les données nettoyées et prétraitées dans un format approprié pour une utilisation future.

"""
    st.markdown(cleaning)


    
