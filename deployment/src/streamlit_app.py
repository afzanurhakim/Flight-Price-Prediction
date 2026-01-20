import streamlit as st
import eda
import prediction

page = st.sidebar.selectbox('Pilih Halaman: ', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.main()
else:
    prediction.main()