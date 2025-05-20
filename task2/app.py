
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# Config
st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️")

MODEL_DIR = "task2"

def cuisine_tokenizer(text):
    return text.split(", ")

@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
    df = joblib.load(os.path.join(MODEL_DIR, "data.pkl"))
    return vectorizer, scaler, pca, kmeans, df

vectorizer, scaler, pca, kmeans, df = load_models()

all_cuisines = sorted({c.strip() for cuisines in df['Cuisines'] for c in cuisines.split(',')})

cuisine_matrix = vectorizer.transform(df['Cuisines'])
cuisine_reduced = pca.transform(cuisine_matrix.toarray())
numeric_features = scaler.transform(df[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']])
combined_features = np.hstack([cuisine_reduced, numeric_features])

def get_user_vector(user_cuisines, user_price, user_rating):
    user_cuisines_str = ", ".join(user_cuisines).lower()
    user_cuisine_vec = vectorizer.transform([user_cuisines_str])
    user_cuisine_red = pca.transform(user_cuisine_vec.toarray())
    user_numeric = scaler.transform([[user_price, user_price, user_rating, 100]])  # Votes as avg
    return np.hstack([user_cuisine_red, user_numeric])

def recommend(user_cuisines, user_price, user_rating, top_n=5):
    user_vector = get_user_vector(user_cuisines, user_price, user_rating)
    user_cluster = kmeans.predict(user_vector)[0]
    cluster_indices = np.where(df['Cluster'] == user_cluster)[0]
    cluster_features = combined_features[cluster_indices]
    similarities = cosine_similarity(user_vector, cluster_features).flatten()
    top_idx = similarities.argsort()[::-1][:top_n]
    recommendations = df.iloc[cluster_indices[top_idx]][['Restaurant Name', 'Cuisines', 'Aggregate rating', 'Price range']]
    # Add similarity scores
    recommendations = recommendations.copy()
    recommendations['Similarity'] = similarities[top_idx]
    return recommendations

st.title("🍽️ Restaurant Recommendation System")
st.write("Select your preferences below to get personalized restaurant suggestions:")

with st.expander("Dataset Summary"):
    st.write(f"Total restaurants in dataset: {len(df)}")
    st.write(f"Unique cuisines available: {len(all_cuisines)}")

cuisine_input = st.multiselect("Preferred cuisines:", options=all_cuisines)
price_input = st.slider("Preferred price range (1=Low, 4=High):", 1, 4, 2)
rating_input = st.slider("Minimum average rating:", 0.0, 5.0, 3.5, 0.1)
top_n = st.slider("Number of recommendations:", 1, 20, 5)

if st.button("Recommend"):
    if not cuisine_input:
        st.warning("Please select at least one cuisine.")
    else:
        recommendations = recommend(cuisine_input, price_input, rating_input, top_n)
        if recommendations.empty:
            st.info("No matching restaurants found.")
        else:
            def stars(rating):
                full = int(rating)
                half = 1 if rating - full >= 0.5 else 0
                return "⭐" * full + ("✨" if half else "")

            results = recommendations.copy()
            results['Aggregate rating'] = results['Aggregate rating'].apply(stars)
            results['Similarity'] = results['Similarity'].apply(lambda x: f"{x:.2f}")
            st.dataframe(results.reset_index(drop=True), use_container_width=True)

            st.markdown("### Explanation:")
            st.write(
                "Restaurants are recommended based on similarity to your chosen cuisines, "
                "price range, and minimum rating. Similarity score shows how closely a restaurant "
                "matches your preferences (1.0 = perfect match)."
            )
if st.button("Reset"):
    st.rerun()


st.markdown("<br><small>🔍 Powered by TF-IDF + PCA + KMeans clustering</small>", unsafe_allow_html=True)