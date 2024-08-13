import streamlit as st
from ..FailuresDetectModel import clean_data
from ..functions import dataframing_data


def page_3():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#3 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Feature Engineering phase.

"""
    st.markdown(texte)

    dataframes = dataframing_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    ## #Before featuring_
        """)
        for name, dataframe in dataframes.items():

            df = clean_data(dataframe)

            st.markdown(f"### {name}")
            st.dataframe(df)


    with col2:
        st.markdown("""
    ## #After featuring_
        """)
        for name, dataframe in dataframes.items():

            df = clean_data(dataframe)
            df = add_features(df)

            st.markdown(f"### {name}")
            st.dataframe(df)
    
