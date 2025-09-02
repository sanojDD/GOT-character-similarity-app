# GOT-character-similarity-app
A simple Streamlit app to explore and compare Game of Thrones characters based on their dialogue embeddings using t-SNE and Euclidean similarity.

![App Screenshot](similarity.png)

<p align="center">
  <img src="similarity.png" width="600"/>
</p>




# Game of Thrones Character Similarity App

This is a simple Streamlit web app that allows users to explore and compare **Game of Thrones** characters based on the similarity of their dialogues. It uses t-SNE for dimensionality reduction and Euclidean distance for similarity comparison.

## 🔍 Features

- Select any character from the show
- View the top 2 most similar characters based on their dialogue usage
- Built with Streamlit and scikit-learn
- Preprocessed data stored in a pickle file

## 📦 Dependencies

Make sure to install the required libraries:

```bash
pip install streamlit pandas scikit-learn

fell free to update this .
make api calls to display images instead of the names only...
