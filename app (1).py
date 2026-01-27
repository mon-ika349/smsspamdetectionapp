import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already downloaded in the environment)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Now import stopwords, as the data should be available
from nltk.corpus import stopwords

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
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("Vectorizer file 'vectorizer.pkl' not found. Please ensure it is available.")
    st.stop()

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is available.")
    st.stop()

# Streamlit app interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
  if input_sms:
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
  else:
    st.warning("Please enter a message to predict.")
