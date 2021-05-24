import torch
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from transformers import AutoTokenizer, AutoModel

@st.cache
def load_assets():
    """
    Load the tokenizer and BERT pretrained model from HuggingFace.
    """

    tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
    model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
    
    return tokenizer, model

def load_classifier():
    return load_model('classifier')

def get_sentiment(text):
    """
    Tokenize a string. Run it through both the pretrained BERT model and the classifier.
    Return a formatted string with the sentiment classification and confidence.
    """
    max_len = 180
    tokens = tokenizer.encode(text, add_special_tokens=True)[:max_len]
    padded = np.array(tokens + [0]*(max_len-len(tokens)))
    attention_mask = np.where(padded != 0, 1, 0)

    padded = np.reshape(padded, newshape=(1,-1))
    attention_mask = np.reshape(attention_mask, newshape=(1,-1))
    
    input_tensor = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        bert_output = model(input_tensor, attention_mask=attention_mask)
        
    features = np.array(bert_output[0][:,0,:])
    sample_prediction = classifier.predict(features)

    if sample_prediction.flatten()[0] > 0.5:
        return f'Positive sentiment! 😃 Model output was {sample_prediction.flatten()[0]}.'
    else:
        return f'Negative Sentiment. 😔 Model output was {sample_prediction.flatten()[0]}.'
        
tokenizer, model = load_assets()
classifier = load_classifier()

with st.form(key='my_form'):
	text = st.text_input(label='Enter some text')
	submit = st.form_submit_button(label='Submit')
    
if submit:
    output = get_sentiment(text)
    st.write(output)