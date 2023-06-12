import streamlit as st
import tensorflow as tf
from transformers import pipeline
from io import StringIO
import gdown
import os


st.set_page_config(page_title="Text Summarizer", page_icon=":rocket:", layout="wide", initial_sidebar_state="auto")

model_url = "https://drive.google.com/drive/folders/1c_FjaIMT48yspZgGj3sVPnwNJTdchUuM"
token_url = "https://drive.google.com/drive/folders/1IP5awf5Zpo3KWTnzk56zNiUTsWrr-9c3"

@st.cache_data
def load_files():
    with st.spinner('downloading tokenizer'):
        gdown.download_folder(token_url, quiet=True, use_cookies=False)
        with st.spinner("downloading model"):
            gdown.download_folder(model_url, quiet=True, use_cookies=False)
    

load_files()

st.write("ALL FILES DOWNLOADED!")


token_path = "first_tokenizer\\"
model_path = "first_model\\"

@st.cache_resource
def load_model(model_path=model_path, tokenizer_path=token_path):
    model = pipeline("summarization", tokenizer=tokenizer_path, model=model_path)
    return model

model = load_model()

st.write("MODEL LOADED!")

st.header("\U0001F4DD Text Summarizer")

text_area = st.text_area('Paste text here', height=500)

st.write("CHECKPOINT 1")

with st.sidebar:
    st.header('Advanced settings')
    st.write("To control the size of the summarized text, toggle the slider below")
    min_token = st.slider('Minimum summary length', min_value=10, max_value=50, value=30)
    max_token = st.slider('Maximum summary length', min_value=70, max_value=300, value=100)
    st.write(' ')
    st.markdown('Made by [Jeremiah Chinyelugo](https://linkedin.com/in/jeremiah-chinyelugo)')
    st.write(' ')
    st.markdown('Source code can be found [here](https://github.com/Jeremyugo/Text-Summarization-BART-Large)')

if st.button('Summarize'):
    if len(text_area) < 1:
        st.write('Please paste a text')
    else:
        st.subheader('Summarized text:')
        text = text_area

        with st.spinner(text="Summarization in progress"):
            
            st.write("CHECKPOINT 2")
            
            output = model(text, min_length=min_token, max_length=max_token)
            generated_summary = output[0]['summary_text']
            
            st.write(generated_summary)


    

    

    

