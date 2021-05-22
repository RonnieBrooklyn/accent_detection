import streamlit as st


st.title("This is a quick demo of The Accent Recognition App")

st.text("Audio samples to be tested")

st.text("American Accent")
st.audio('./english33.wav')

st.text("British Accent")
st.audio('./english38.wav')

st.text("Australian Accent")
st.audio('./english77.wav')


st.subheader("Please select a test file from below to run with the accent app..")
option = st.selectbox('Select', ["American",'British','Australian'])
st.write('You Selected: ', option)

if st.button('Start Testing'):
    st.write('Test in progress..')
else:
    st.write('Test not yet started')