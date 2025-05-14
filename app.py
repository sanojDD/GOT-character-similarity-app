import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import euclidean_distances

# Load data from the pickle file
with open("data.pkl", "rb") as f:
    df = pickle.load(f)

st.title("Game of Thrones Character Similarity")

# Character selection
character_list = df["character"].unique()
selected_char = st.selectbox("Select a character:", character_list)

# Get selected character's coordinates
selected_row = df[df["character"] == selected_char]
selected_coords = selected_row[["x", "y"]].values

# Compute Euclidean distance to all other characters
df["distance"] = euclidean_distances(df[["x", "y"]], selected_coords).flatten()

# Exclude the selected character and get top 5 similar
similar_df = df[df["character"] != selected_char].sort_values(by="distance").head(2)

# Display result
st.subheader("Top 5 similar characters:")
st.dataframe(similar_df[["character", "distance"]])
