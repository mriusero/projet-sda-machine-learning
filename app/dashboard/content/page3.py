import streamlit as st
from ..FailuresDetectModel import clean_data
from ..functions import DataVisualizer
from ..FailuresDetectModel import FeatureAdder, run_statistical_test

def page_3():

    st.markdown('<div class="header">#3 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Feature Engineering phase.

"""
    st.markdown(texte)
    st.markdown(f"## ---------------------------------------- Linear Regression ----------------------------------------")
    st.write('### #FeatureAdder')
    col1, col2 = st.columns(2)
    with col1:
        st.write('### #Train_df')
        st.session_state.data.distribution_histogram('train', 'time (months)', 'blue')
        #st.dataframe(train_df)

        #st.write(run_statistical_test(train_df, 'normality', 'length_filtered' ))
        #st.write(run_statistical_test(train_df, 'normality', 'length_measured'))

#    with col2:
#        fig = display_train.plot_distribution_histogram('time (months)')
#        st.plotly_chart(fig)
#
#        stats_df = train_df.groupby('item_index').agg({'time (months)': ['max', 'min', 'mean', 'std']})
#        display_stats = DisplayData(stats_df)
#        st.dataframe(stats_df)
#        st.plotly_chart(display_stats.plot_distribution_histogram('time (months)'))
#
#
#    st.write('### #Model preprocessor_')
#    col1, col2 = st.columns(2)
#    with col1:
#        st.write('### #Train_df')
#        st.dataframe(train_df)
#    with col2:
#        st.write('### #Test_df')
#        st.dataframe(test_df)




    st.markdown(f"## ---------------------------------------- LSTM Model ----------------------------------------")

    st.markdown(f"## --------------------------------- Random Forest Classifier -----------------------------------")
