import streamlit as st
import pickle
import re
import tweepy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# --- Setup ---  streamlit run d:\tw_project\ppp\app2.py [ARGUMENTS]
nltk.download('stopwords')

# Get your Bearer Token from Twitter Developer Portal
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFZc2wEAAAAAMO7nrZwnL970TEhmWKW7uU1F7Nw%3DezWyIrqvlXCqHgLVifdcSAEH4wZPrQTnkb9meAp0CTS9uSuayk"  # Replace with your actual token

#AAAAAAAAAAAAAAAAAAAAAFZc2wEAAAAAMO7nrZwnL970TEhmWKW7uU1F7Nw%3DezWyIrqvlXCqHgLVifdcSAEH4wZPrQTnkb9meAp0CTS9uSuayk
# --- Load Stopwords ---
@st.cache_resource
def load_stopwords():
    return stopwords.words('english')

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# --- Preprocess & Predict Sentiment ---
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    vectorized = vectorizer.transform([text])
    sentiment = model.predict(vectorized)
    return "Negative" if sentiment == 0 else "Positive"

# --- Create Visual Sentiment Card ---
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    return f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """

# --- Fetch Tweets via Twitter API ---
def fetch_user_tweets(username, max_results=5):
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN)
        user = client.get_user(username=username)
        user_id = user.data.id
        tweets = client.get_users_tweets(id=user_id, max_results=max_results, tweet_fields=["text"])
        return [tweet.text for tweet in tweets.data] if tweets.data else []
    except Exception as e:
        st.error(f"Twitter API error: {e}")
        return []

# --- Streamlit App ---
def main():
    st.title("📊 Twitter Sentiment Analysis")
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch Tweets"):
            if username.strip():
                tweets = fetch_user_tweets(username)
                if not tweets:
                    st.info("No tweets found or user is private.")
                else:
                    for tweet_text in tweets:
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter a username.")

if __name__ == "__main__":
    main()

