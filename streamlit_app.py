import streamlit as st 
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from PIL import Image
nltk.download('punkt')
nltk.download('stopwords')

image = Image.open('me2.png')

st.image(image, caption='EMAIL')



def transform_text(text):
    text = text.lower()
    
    text= nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and  i not in string.punctuation:
            y.append(i)
            
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email ", "Via SMS", "other"])


if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]


    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')


