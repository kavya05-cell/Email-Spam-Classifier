import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources only if missing
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}")  # stopwords lives here
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{pkg}")  # punkt & punkt_tab live here
        except LookupError:
            nltk.download(pkg)

ps = PorterStemmer()

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

# Load vectorizer & model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📧 Email Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam!!")
    else:
        st.header("Not Spam")


