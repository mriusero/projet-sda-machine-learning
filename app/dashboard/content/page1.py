import streamlit as st

from ..components import plot_crack_length_by_item_with_failures
def page_1(dataframes):
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#1 Input Analysis_</div>', unsafe_allow_html=True)    
    tree = """

"""
    st.markdown(tree)      
    
    training_description = """

"""
    st.markdown(training_description)

        # Exemple d'utilisation
    fig1 = plot_crack_length_by_item_with_failures(dataframes['train'])
    st.pyplot(fig1)