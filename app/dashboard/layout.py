import os
import streamlit as st

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app_layout():
    from .content import page_0, page_1, page_2, page_3, page_4, page_5, page_6

    st.set_page_config(
        page_title="SDA-MACHINE-LEARNING",
        page_icon="",
        layout='wide',
        initial_sidebar_state="auto",
        menu_items={
            'About': "#Github Repository :\n\nhttps://github.com/mriusero/projet-sda-machine-learning/blob/main/README.md"
        }
    )

    load_css()
    page = st.sidebar.radio("Overview", ["#0 Introduction_",
                                         "#1 Exploration_",
                                         "#2 Cleaning_",
                                         "#3 Feature Engineering_",
                                         "#4 Statistics & Preprocessing_",
                                         "#5 Training_",
                                         "#6 Prediction_",
                                         ])

    if page == "#0 Introduction_":
        page_0()
    elif page == "#1 Exploration_":
        page_1()
    elif page == "#2 Cleaning_":
        page_2()
    elif page == "#3 Feature Engineering_":
        page_3()
    elif page == "#4 Statistics & Preprocessing_":
        page_4()
    elif page == "#5 Training_":
        page_5()
    elif page == "#6 Prediction_":
        page_6()








