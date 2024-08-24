import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
import base64

# Convert local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_string}"

# Set background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('C:/Users/91762/Desktop/Books Recommendation/books.csv', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
    df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
    df = df.dropna(subset=['average_rating', 'ratings_count'])

    df['rating_between'] = pd.cut(
        df['average_rating'],
        bins=[-float('inf'), 1, 2, 3, 4, 5, float('inf')],
        labels=['below 1', '1 to 2', '2 to 3', '3 to 4', '4 to 5', 'above 5']
    )

    rating_df = pd.get_dummies(df['rating_between'])
    language_df = pd.get_dummies(df['language_code'], prefix='language')

    features = pd.concat([
        rating_df,
        language_df,
        df[['average_rating', 'ratings_count']]
    ], axis=1)

    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)

    return df, features

def fit_model(features):
    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)
    return model

def recommend_books(book_name, df, model, features):
    matches = df['title'].str.contains(book_name, case=False, na=False)
    if not matches.any():
        return f"Book '{book_name}' not found in the dataset."

    book_id = df[matches].index[0]
    dist, idlist = model.kneighbors(features)
    
    book_list_name = []
    for newid in idlist[book_id]:
        book_list_name.append(df.loc[newid, 'title'])
    
    return book_list_name

def main():
    st.title("Book Recommendation System")

    # Set background image
    image_path = 'C:/Users/91762/Downloads/OIP (8).jpeg'
    image_url = get_base64_image(image_path)
    set_background(image_url)

    # Load and preprocess data
    df = load_data()
    if df is None:
        return
    df, features = preprocess_data(df)
    model = fit_model(features)

    # Display available book titles
    st.write("Available book titles:")
    st.write(df['title'].sample(10).to_list())  # Display a sample of titles

    # User input
    book_name = st.text_input("Enter the book title:", "")

    if book_name:
        recommendations = recommend_books(book_name, df, model, features)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Recommended Books:")
            for i, book in enumerate(recommendations):
                st.write(f"{i + 1}. {book}")

if __name__ == "__main__":
    main()
