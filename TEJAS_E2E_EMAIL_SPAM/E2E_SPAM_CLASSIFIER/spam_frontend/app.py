import streamlit as st
import json
import requests

url = "http://backend.docker:8000/predict"
header = {'Content-Type': 'application/json'}

st.set_page_config(page_title="Spam Classifier Front End")
st.title("Welcome to the Spam Classifier Dashboard")
st.write("Enter the Email you want to Predict")

text = st.text_input("Email to predict")
if text and st.button('Spam Classifier Prediction'):  # Check if text is not empty and button is pressed
    data = {"text": text}
    payload = json.dumps(data)
    response = requests.request("POST", url, headers=header, data=payload)
    response = response.text    
    st.write(f"The Email is {str(response)}")
elif not text:  # Handle case when text is empty
    st.write("Please enter an email to predict.")
