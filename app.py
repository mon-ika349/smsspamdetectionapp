import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download NLTK data (if not already downloaded in the environment)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True) # Explicitly download punkt_tab

# Initialize PorterStemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

# Load the pre-trained TfidfVectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
  # Preprocess the input SMS
  transformed_sms = transform_text(input_sms)
  # Vectorize the preprocessed SMS
  vector_input = tfidf.transform([transformed_sms])
  # Make prediction
  result = model.predict(vector_input)[0]

  # Display result
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")
