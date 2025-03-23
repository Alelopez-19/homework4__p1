import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate, Input
import streamlit as st

# Load and Prepare Data
def load_and_prepare_data():
    df = pd.read_csv("NintendoGames.csv", parse_dates=['date'])
    df['meta_score'].fillna(df['meta_score'].mean(), inplace=True)
    
    # Label encoding for Platform and Developer
    platform_encoder = LabelEncoder()
    df['Platform_encoded'] = platform_encoder.fit_transform(df['platform'])
    print(platform_encoder.classes_)  # To check the encoded classes for platforms
    
    developer_encoder = LabelEncoder()
    df['Developer_encoded'] = developer_encoder.fit_transform(df['developers'])
    
    # Genres Processing
    df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    df = pd.concat([df, genre_df], axis=1)
    
    # Features and Target
    X_platform = df['Platform_encoded'].values
    X_developer = df['Developer_encoded'].values
    X_genres = df[mlb.classes_].values
    y = df['meta_score'].values
    
    return df, X_platform, X_developer, X_genres, y, platform_encoder, developer_encoder, mlb

# Train Neural Network Model
def train_nn_model(X_train_platform, X_train_dev, X_train_genres, y_train, X_test_platform, X_test_dev, X_test_genres, y_test, platform_encoder, developer_encoder, mlb):
    num_platforms = len(platform_encoder.classes_)
    num_developers = len(developer_encoder.classes_)
    num_genres = X_train_genres.shape[1]

    # Inputs and Embedding layers
    platform_input = Input(shape=(1,))
    developer_input = Input(shape=(1,))
    genres_input = Input(shape=(num_genres,))

    platform_embedding = Embedding(input_dim=num_platforms, output_dim=5)(platform_input)
    developer_embedding = Embedding(input_dim=num_developers, output_dim=10)(developer_input)
    
    platform_flat = Flatten()(platform_embedding)
    developer_flat = Flatten()(developer_embedding)
    
    merged = Concatenate()([platform_flat, developer_flat, genres_input])
    
    x = Dense(64, activation='relu')(merged)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    
    model = keras.Model(inputs=[platform_input, developer_input, genres_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit([X_train_platform, X_train_dev, X_train_genres], y_train, validation_data=([X_test_platform, X_test_dev, X_test_genres], y_test), epochs=50, batch_size=32)
    
    return model

# Display Results
def display_results(df):
    # Box Plot by Genre
    fig = px.box(df.explode('genres'), x="genres", y="meta_score", title="Meta Score Distribution by Genre")
    st.plotly_chart(fig)

    # Average Meta Score by Platform
    avg_score_per_platform = df.groupby("platform")["meta_score"].mean().reset_index()
    fig = px.bar(avg_score_per_platform, x="platform", y="meta_score", title="Average Meta Score by Platform")
    st.plotly_chart(fig)

def main():
    # Load and prepare data
    df, X_platform, X_developer, X_genres, y, platform_encoder, developer_encoder, mlb = load_and_prepare_data()

    # Split data (train/test)
    X_train_platform, X_test_platform, X_train_dev, X_test_dev, X_train_genres, X_test_genres, y_train, y_test = train_test_split(
        X_platform, X_developer, X_genres, y, test_size=0.2, random_state=42)

    # Ensure no out-of-range indices
    assert np.all(X_train_platform < len(platform_encoder.classes_)), "Some platform indices are out of range"
    assert np.all(X_test_platform < len(platform_encoder.classes_)), "Some platform indices are out of range"
    assert np.all(X_train_dev < len(developer_encoder.classes_)), "Some developer indices are out of range"
    assert np.all(X_test_dev < len(developer_encoder.classes_)), "Some developer indices are out of range"
    
    # Train neural network model
    model = train_nn_model(X_train_platform, X_train_dev, X_train_genres, y_train, X_test_platform, X_test_dev, X_test_genres, y_test, platform_encoder, developer_encoder, mlb)

    # Display results
    display_results(df)

if __name__ == "__main__":
    main()
