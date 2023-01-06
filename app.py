import streamlit as st
import pred
import pickle

model = pickle.load(open('grid.pkl', 'rb'))

st.header('Similar or Not')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = pred.data_point(q1, q2)
    result = model.predict(query)[0]
    
    st.header('Similar') if result else st.header('Unique')

