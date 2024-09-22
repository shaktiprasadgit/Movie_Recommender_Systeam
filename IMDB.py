import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your IMDB data for the last 10 years (example structure)
@st.cache_data
def load_data():
    # Load your dataset here, filtered to the last 10 years
    df = pd.read_csv('D:\Data Science\CSV Files/IMDB-Movie-Data.csv')  # Make sure this data is prefiltered for 2014-2024
    return df

# Preprocess the dataset to merge genres, directors, actors as a text field
def preprocess_data(df):
    df['combined_features'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Actors']
    return df

# Function to get recommendations based on the selected movie
def recommend_movies(movie, similarity_matrix, df):
    idx = df[df['Title'] == movie].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [i[0] for i in similarity_scores[1:6]]  # Recommend top 5 movies
    return df['Title'].iloc[recommended_movie_indices]

# Streamlit app layout
st.title('Movie Recommender System')

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

# TF-IDF Vectorizer to compute similarity based on combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Select a movie to get recommendations
movie_list = df['Title'].values
selected_movie = st.selectbox("Select a movie to get recommendations", movie_list)

if st.button('Recommend'):
    recommendations = recommend_movies(selected_movie, cosine_sim, df)
    st.write('Here are the top 5 recommendations:')
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")

