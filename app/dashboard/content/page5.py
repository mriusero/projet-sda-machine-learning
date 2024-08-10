import streamlit as st

# Initialisation des configurations des modèles
MODEL_COLUMN_CONFIG = {
    'RandomForestModel': {
        'X': ['crack length (arbitary unit)', 'time (months)'],
        'y': 'label'
    },
    'LogisticRegressionModel': {
        'X': ['crack length (arbitary unit)', 'time (months)'],
        'y': 'label'
    }
}
def update_model_config(model_name, x_columns, y_column):
    """
    Update model config.
    """
    MODEL_COLUMN_CONFIG[model_name] = {
        'X': x_columns,
        'y': y_column
    }

def page_5():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)  # TITLE
    st.markdown('<div class="header">#5 Training_</div>', unsafe_allow_html=True)
    texte = """
    Here is the Training phase.
    """
    st.markdown(texte)

    st.markdown('<div class="header">Model Configuration</div>', unsafe_allow_html=True)

    model_names = list(MODEL_COLUMN_CONFIG.keys())
    selected_model = st.selectbox("Choisissez un modèle", model_names)

    current_config = MODEL_COLUMN_CONFIG[selected_model]

    st.subheader("Configurer les colonnes X")
    x_columns = st.text_area(
        "Entrez les noms des colonnes X, séparés par une virgule",
        value=", ".join(current_config['X'])
    )
    x_columns = [col.strip() for col in x_columns.split(",")]

    st.subheader("Configurer la colonne Y")
    y_column = st.text_input("Entrez le nom de la colonne Y", value=current_config['y'])

    if st.button("Sauvegarder la configuration"):
        update_model_config(selected_model, x_columns, y_column)
        st.success(f"Configuration mise à jour pour {selected_model}!")

    st.subheader("Configuration actuelle")
    st.write(MODEL_COLUMN_CONFIG)

