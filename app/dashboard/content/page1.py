import streamlit as st
from ..functions import load_data, merge_data
from ..components import plot_crack_length_by_item_with_failures
def page_1():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#1 Input Analysis_</div>', unsafe_allow_html=True)    
    tree = """

"""
    st.markdown(tree)      
    
    training_description = """

"""
    st.markdown(training_description)

    training_data = load_data()
    train, pseudo_test, pseudo_test_with_truth, test = merge_data(training_data)

    # Exemple d'utilisation
    fig1 = plot_crack_length_by_item_with_failures(train)
    st.pyplot(fig1)