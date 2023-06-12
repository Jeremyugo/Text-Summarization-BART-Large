import streamlit as st
import tensorflow as tf
from transformers import pipeline
from io import StringIO

model_path = ".\\results\\tuned_model"
tokenizer_path = ".\\results\\tuned_tokenizer"

@st.cache_resource
def load_model(model_path=model_path, tokenizer_path=tokenizer_path):
    model = pipeline("summarization", tokenizer=tokenizer_path, model=model_path)
    return model

model = load_model(model_path, tokenizer_path)

st.header("\U0001F4DD Text Summarizer")

text_area = st.text_area('Paste text here', height=500)

with st.sidebar:
    st.header('Advanced settings')
    st.write("To control the size of the summarized text, toggle the slider below")
    min_token = st.slider('Minimum summary length', min_value=10, max_value=50, value=30)
    max_token = st.slider('Maximum summary length', min_value=70, max_value=300, value=100)
    st.write(' ')
    st.markdown('Made by [Jeremiah Chinyelugo](https://linkedin.com/in/jeremiah-chinyelugo)')
    st.write(' ')
    st.markdown('Source code can be found [here](https://github.com/Jeremyugo)')

if st.button('Summarize'):
    if len(text_area) < 1:
        st.write('Please paste a text')
    else:
        st.subheader('Summarized text:')
        text = text_area

        with st.spinner(text="Summarization in progress"):
            output = model(text, min_length=min_token, max_length=max_token)
            generated_summary = output[0]['summary_text']
            
            st.write(generated_summary)


    

    

    

