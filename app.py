import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = joblib.load('best_logistic_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess text
def preprocess_text(text):
     # Remove special characters, numbers, and links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)


# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    text = preprocess_text(text)
    # Vectorize the text using TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])
    # Predict sentiment using the trained model
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Streamlit app
def main():
    st.title('Product Sentiment Analysis')
    st.write('Enter your product review below:')
    
    # Text input for user input
    user_input = st.text_area('Input your review here:')
    
    if st.button('Predict Sentiment'):
        if user_input:
            # Predict sentiment
            prediction = predict_sentiment(user_input)
            # Display the prediction
            if prediction == 'positive':
                st.success('Sentiment: Positive')
            elif prediction == 'negative':
                st.error('Sentiment: Negative')
            else:
                st.warning('Sentiment: Neutral')
        else:
            st.warning('Please enter a review before predicting.')

if __name__ == '__main__':
    main()
