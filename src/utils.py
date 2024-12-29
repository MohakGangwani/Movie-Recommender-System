import os
import pandas as pd
import yaml
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
import pickle
from pathlib import Path

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
    Saves preprocessed data to a CSV for future use.
    """
    preprocessed_path = params["preprocessed_data_path"]
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}")
        return pd.read_pickle(preprocessed_path)
    else:
        print(f"Preprocessing raw data from {params['data_path']}")
        raw_data = pd.read_csv(params["data_path"])
        preprocessed_data = preprocess_data(raw_data, params)
        Path("/".join(preprocessed_path.split("/")[:-1])+"/").mkdir(parents=True, exist_ok=True)
        pd.to_pickle(preprocessed_path, index=False)
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
        Path("/".join(vector_path.split("/")[:-1])+"/").mkdir(parents=True, exist_ok=True)
        with open(vector_path, "wb") as vector_file:
            pickle.dump(vector, vector_file)
        return vector


def preprocess_large_text(texts):
    """
    Preprocess a list of large texts using spaCy, including lemmatization and stopword removal.
    """
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(
        texts, batch_size=100, n_process=os.cpu_count(), disable=["ner", "parser"]
    )
    return [
        " ".join(
            [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        )
        for doc in docs
    ]


def preprocess_data(data, params):
    """
    Preprocess the dataset by cleaning, lemmatizing, and creating tags.
    Returns a preprocessed DataFrame.
    """
    data.dropna(subset=["title"], inplace=True)
    drop_cols = params["drop_columns"]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Clean text fields
    fields_to_clean = [
        "production_companies",
        "production_countries",
        "spoken_languages",
        "cast",
        "director",
        "writers",
        "producers",
        "genres",
    ]
    for field in fields_to_clean:
        if field in data.columns:
            data[field] = data[field].str.replace(" ", "").str.replace(",", " ", regex=True)

    # Generate combined Tags
    data["Tags"] = (
        data["title"].fillna("")
        + " " + data["overview"].fillna("")
        + " " + data["genres"].fillna("")
        + " " + data["production_companies"].fillna("")
        + " " + data["production_countries"].fillna("")
        + " " + data["spoken_languages"].fillna("")
        + " " + data["cast"].fillna("")
        + " " + data["director"].fillna("")
        + " " + data["writers"].fillna("")
        + " " + data["producers"].fillna("")
    )
    data = data[["title", "overview", "Tags", "poster_path"]]
    data["Tags"] = data["Tags"].astype(str).str.replace(r"[^\p{L}\s]", "").str.lower()
    data["Processed_Tags"] = preprocess_large_text(data["Tags"])
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
        print("Creating similarity model")
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(vector)
        Path("/".join(model_path.split("/")[:-1])+"/").mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as model_file:
            pickle.dump(nn, model_file)
        return nn


def recommend_movie(movie_id, params):
    vector = get_vector(None, params)[movie_id]
    model = get_sim_model(vector, params)
    movies = model.kneighbors(vector, n_neighbors=params["nrecommendatios"]+1, return_distance=False).flatten()
    data = pd.read_pickle(params["preprocessed_data_path"])
    movies = [
        (
            data.iloc[id,"title"], 
            ("https://image.tmdb.org/t/p/original/"+data.iloc[id,"poster_path"])
        ) 
        for id in movies[1:]
        ]
    description = data.iloc[movies[0],"overview"]
    return movies, description