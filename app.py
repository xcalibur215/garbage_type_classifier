import streamlit as st
import requests
from PIL import Image
import io

st.title('Garbage Type Classifier')

st.write('Upload one or more images to classify their garbage type (cardboard, glass, metal, paper, plastic, trash)')

uploaded_files = st.file_uploader('Choose image(s)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    files = [('files', (f.name, f, f.type)) for f in uploaded_files]
    response = requests.post('http://localhost:8000/predict/', files=files)
    if response.status_code == 200:
        results = response.json()['results']
        for i, result in enumerate(results):
            st.image(uploaded_files[i], caption=f"Prediction: {result['class']}", use_container_width=True)
            st.write(f"Filename: {result['filename']}")
            st.write(f"Predicted Class: {result['class']}")
    else:
        st.error('Prediction failed. Please check the FastAPI server.')
