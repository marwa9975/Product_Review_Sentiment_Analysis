import streamlit as st
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)

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
    text_vectorized = loaded_tfidf_vectorizer.transform([text])
    # Predict sentiment using the trained model
    prediction = loaded_model.predict(text_vectorized)
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
