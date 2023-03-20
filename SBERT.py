
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:30:35 2023

@author: Amitesh
"""
#working perfectly

import spacy
import streamlit as st
import string
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA



st.cache(allow_output_mutation=True)
from sentence_transformers import SentenceTransformer,util
loaded_model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

lr_model = joblib.load('LRmodel.pkl')

with st.sidebar:
    
    selected = option_menu('Texton',
                          
                          ['Similarity Checker',
                           'Text Classification',
                           'Similarity Checker Advance'],
                          icons=['bookmark-check-fill','emoji-angry','list-columns-reverse'],
                          default_index=0)
    


def spacy_tokenizer(sentence):
    doc = nlp(sentence)
    mytokens = [ word.lemma_.lower().strip() for word in doc ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    sentence = " ".join(mytokens)
    return sentence

def text_classification(input_data):
    token = spacy_tokenizer(input_data)
    arr = loaded_model.encode(token)
    pred = lr_model.predict(arr.reshape(1,-1))
    
    if (pred == 1):
      return 'The Comment is Toxic'
    else:
      return 'The Comment is not Toxic'
  
def word_sim(input_data):
    
    embeddings = loaded_model.encode(input_data)
    cos_sim = util.cos_sim(embeddings, embeddings)
    
    return cos_sim
  
# Similarity checker
if (selected == 'Similarity Checker'):
    
    # page title
    st.title('Sentence Similarity Checker')
    
    
    col1, col2 = st.columns(2)
    # getting the input data from the user
    
    with col1:
        word_1 = st.text_input('First Input','I am riding a horse')
        
    with col2:
        word_2 = st.text_input('Second Input','I am riding a white horse')
    
    
    # code for Prediction
    sim = ''

    # creating a button for Prediction

    if st.button('Check Similarity'):
        sim = word_sim([word_1,word_2])
        b = sim[0][1]
        sim = f'Similarity between these two words are {b*100} %'
    st.success(sim)
    
if (selected == 'Text Classification'):
    
    #pagetitle
    st.title('Text Classification')
    st.markdown('This will classify wheather your text is **:red[toxic]** or **:green[not toxic]**')
    text_1 = st.text_input('Enter the Sentence','My favourite color is Green')
    output = ''
    if st.button('Text Classification result'):
        output = text_classification(text_1)
    st.success(output)
        
        
        
        
# Similarity checker
if (selected == 'Similarity Checker Advance'):
    st.markdown("You will get better results if you provide 9 inputs")
    
    # page title
    st.title('Advance Similarity Checker')
    
    
    col1, col2 , col3 = st.columns(3)
    # getting the input data from the user
    
    with col1:
        word_1 = st.text_input('First Input')
        
    with col2:
        word_2 = st.text_input('Second Input')
        
    with col3:
        word_3 = st.text_input('Third Input')
    
    with col1:
        word_4 = st.text_input('Fourth Input')
        
    with col2:
        word_5 = st.text_input('Fifth Input')
        
    with col3:
        word_6 = st.text_input('Sixth Input')
        
    with col1:
        word_7 = st.text_input('Seventh Input')
        
    with col2:
        word_8 = st.text_input('Eighth Input')
        
    with col3:
        word_9 = st.text_input('Ninth Input')
    
    
    # code for Prediction
    sim = ''

    # creating a button for Prediction

    if st.button('Check Similarity'):
        sentences = [word_1,word_2,word_3,word_4,word_5,word_6,word_7,word_8,word_9]
        cos_sim = word_sim(sentences)
        cos_sim = cos_sim.detach().numpy()
        pca = PCA(n_components=2)
        result = pca.fit_transform(cos_sim)
        fig2,ax = plt.subplots()
        plt.scatter(result[:,0], result[:,1])
        for i, word in enumerate(sentences):
            plt.annotate(sentences[i], xy=(result[i, 0], result[i, 1]))
        df = pd.DataFrame(cos_sim, index = sentences,
                      columns = sentences)
       # figure(figsize=(12, 10), dpi=80)
        fig, ax = plt.subplots()
        sns.heatmap(df,fmt="",cmap='RdYlGn',linewidths=0.30)
        st.write(fig)
        st.write(fig2)
    st.success(sim)
    
