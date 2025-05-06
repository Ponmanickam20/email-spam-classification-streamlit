import streamlit as st
import joblib
import spacy

# Load the trained model and vectorizer
model = joblib.load('svm_email.pkl')
vectorizer = joblib.load('Vector_email.pkl')
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Preprocess the text using spaCy and vectorizer
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return lemmatized_text

# Streamlit interface
st.title("Email Classification with SVM")
st.write("Enter the email text to classify its sentiment.")

label_mapping = {0: "No Spam", 1: "Spam"}

# Text input
email_text = st.text_area("Email Text", "")

if st.button("Classify"):
    if email_text:
        # Preprocess and vectorize the input text
        preprocessed_text = preprocess_text(email_text)
        text_vector = vectorizer.transform([preprocessed_text])

        # Convert the sparse matrix to a dense array
        text_vector_dense = text_vector.toarray()

        # Predict using the SVM model
        prediction = model.predict(text_vector_dense)[0]

        sentiment_label = label_mapping[prediction]

        # Display the result
        st.write(f"The email is classified as: **{sentiment_label}**")
    else:
        st.write("Please enter some text to classify.")
