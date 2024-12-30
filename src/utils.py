import os
import pandas as pd
import yaml
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
import pickle
from pathlib import Path
from pandarallel import pandarallel
import re

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

# Download stopwords
nltk.download("stopwords")

def install_spacy_model():
    os.system("python -m spacy download en_core_web_sm")

def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)

def load_preprocess_data(params):
    """
    Load preprocessed data if it exists, otherwise preprocess raw data.
    Saves preprocessed data to a Parquet file for future use.
    """
    preprocessed_path = params["preprocessed_data_path"]
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}")
        return pd.read_parquet(preprocessed_path)
    else:
        print(f"Preprocessing raw data from {params['data_path']}")
        raw_data = pd.read_csv(params["data_path"])
        preprocessed_data = preprocess_data_parallel(raw_data, params)
        Path(preprocessed_path).parent.mkdir(parents=True, exist_ok=True)
        preprocessed_data.to_parquet(preprocessed_path, compression="snappy")
        return preprocessed_data

def get_vector(data_preprocessed, params):
    """
    Load or generate a vectorized representation of the preprocessed tags.
    Saves the vector to a file for future use.
    """
    vector_path = params["vector_path"]
    if os.path.exists(vector_path):
        print(f"Loading vector from {vector_path}")
        with open(vector_path, "rb") as vector_file:
            return pickle.load(vector_file)
    else:
        print("Generating vector using CountVectorizer")
        cv = CountVectorizer(
            max_df=params["cv_max_df"],
            min_df=params["cv_min_df"],
            max_features=params["cv_max_features"],
            stop_words="english",
        )
        vector = cv.fit_transform(data_preprocessed["Processed_Tags"])
        Path(vector_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vector_path, "wb") as vector_file:
            pickle.dump(vector, vector_file)
        return vector

def preprocess_large_text(texts):
    """
    Preprocess a list of large texts using spaCy, including lemmatization and stopword removal.
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    docs = nlp.pipe(texts, batch_size=200, n_process=os.cpu_count())
    return [
        " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
        for doc in docs
    ]

def preprocess_data_parallel(data, params):
    """
    Preprocess data with parallelization.
    """
    data.dropna(subset=["title"], inplace=True)
    drop_cols = params["drop_columns"]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Clean text fields in parallel
    fields_to_clean = [
        "production_companies", "production_countries", 
        "spoken_languages", "cast", "director", "writers", 
        "producers", "genres"
    ]
    for field in fields_to_clean:
        if field in data.columns:
            data[field] = data[field].parallel_apply(lambda x: re.sub(r"[^\w\s]", " ", str(x).replace(",", " ")))

    # Generate combined Tags
    data["Tags"] = data.parallel_apply(
        lambda row: " ".join([str(row[field]) for field in fields_to_clean if field in row]), axis=1
    )

    # Preprocess text
    data["Tags"] = data["Tags"].parallel_apply(lambda x: re.sub(r"[^\w\s]", "", x.lower()))
    data["Processed_Tags"] = preprocess_large_text(data["Tags"])

    # Update poster_path column
    base_url = "https://image.tmdb.org/t/p/original"
    data["poster_path"] = data["poster_path"].apply(
        lambda x: base_url + x if pd.notna(x) else params["img_not_available"]
    )

    data.reset_index(drop=True, inplace=True)
    return data

def get_sim_model(vector, params):
    """
    Load or create a similarity model using NearestNeighbors.
    Saves the model to a file for future use.
    """
    model_path = params["model_path"]
    if os.path.exists(model_path):
        print(f"Loading similarity model from {model_path}")
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    else:
        print("Creating Similarity Model")
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(vector)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as model_file:
            pickle.dump(nn, model_file)
        return nn
    

def recommend_movie(movie_id, preloaded_data):
    vector = preloaded_data["vectors"][movie_id]
    model = preloaded_data["model"]
    similar_movie_ids = model.kneighbors(vector, n_neighbors=preloaded_data["params"]["nrecommendations"] + 1, return_distance=False).flatten()
    recommended_movies = {id: preloaded_data["movies"][id] for id in similar_movie_ids[1:]}  # Exclude the input movie
    return recommended_movies