import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Load the movies data
@st.cache_data
def load_data():
    # Replace this URL with your actual data source
    df = pd.read_csv('D:\Data Science\CSV Files/IMDB-Movie-Data.csv')
    
    # Convert 'year' to datetime and filter for last 10 years
    current_year = datetime.datetime.now().year
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df[df['Year'].dt.year >= current_year - 10]
    
    return df

# Create a content-based recommender system
def get_recommendations(title, cosine_sim, df):
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]

# Streamlit app
def main():
    st.title('Movie Recommender System')
    
    # Load data
    df = load_data()
    
    # Create a CountVectorizer object
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['Genre'])
    
    # Compute the Cosine Similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Get list of movie titles
    movie_list = df['Title'].tolist()
    
    # Create a dropdown to select a movie
    selected_movie = st.selectbox('Select a movie you like:', movie_list)
    
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(selected_movie, cosine_sim, df)
        
        st.subheader('Recommended Movies:')
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

if __name__ == "__main__":
    main()