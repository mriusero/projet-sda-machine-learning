import streamlit as st
from ..FailuresDetectModel import clean_data, run_statistical_test
from ..functions import dataframing_data

def page_4():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#4 Statistics & Preprocessing_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Statistics & Preprocessing phase.

"""
    st.markdown(texte)

    dataframes = dataframing_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## #Before statistics_
            """)
        for name, dataframe in dataframes.items():
            df = clean_data(dataframe)

            st.markdown(f"### {name}")
            st.dataframe(df)

    with col2:
        st.markdown("""
        ## #After statistics_
            """)
        for name, dataframe in dataframes.items():
            df = clean_data(dataframe)
            df = add_features(df)

            st.markdown(f"### {name}")
            result = run_statistical_test(df, 'normality', 'time (months)')
            st.write(result)
            result = run_statistical_test(df, 'normality', 'crack length (arbitary unit)')
            st.write(result)
