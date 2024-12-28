import pandas as pd
import numpy as np
import os
import yaml
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_large_text(texts):
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(texts, batch_size=10000, n_process=-1, disable=["ner", "parser"])
    return [
        " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
        for doc in docs
    ]

def preprocess_data(data):
    data.dropna(subset=['title'])
    drop_cols = [
        'id', 
        'vote_average', 
        'vote_count', 
        'status', 
        'release_date', 
        'revenue', 
        'runtime', 
        'budget', 
        'imdb_id', 
        'original_language', 
        'original_title', 
        'popularity', 
        'imdb_rating', 
        'imdb_votes', 
        'tagline', 
        'music_composer', 
        'director_of_photography'
        ]
    data.drop(columns=drop_cols, inplace=True)
    data.production_companies = data.production_companies.str.replace(" ", "").str.replace(",", " ")
    data.production_countries = data.production_countries.str.replace(" ", "").str.replace(",", " ")
    data.spoken_languages = data.spoken_languages.str.replace(" ", "").str.replace(",", " ")
    data.cast = data.cast.str.replace(" ", "").str.replace(",", " ")
    data.director = data.director.str.replace(" ", "").str.replace(",", " ")
    data.writers = data.writers.str.replace(" ", "").str.replace(",", " ")
    data.producers = data.producers.str.replace(" ", "").str.replace(",", " ")
    data.genres = data.genres.str.replace(" ", "").str.replace(",", " ")
    
    data["Tags"] = data.title + " " + data.overview + " " + data.genres + " " + data.production_companies + " " + data.production_countries + " " + data.spoken_languages + " " + data.cast + " " + data.director + " " + data.writers + " " + data.producers
    data = data[["title", "Tags", "poster_path"]]
    data['Tags'] = data['Tags'].astype(str)
    data["Tags"] = data.Tags.str.replace(r"[^\p{L}\s]", "").str.lower()
    data['Processed_Tags'] = preprocess_large_text(data['Tags'])
    return data

def lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text) if w not in stopwords.words('english')]

def fetch_poster(path):
    return "https://image.tmdb.org/t/p/original/"+path


def main():
    global params
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    df = pd.read_csv(params["data_path"])
    df_preprocessed = preprocess_data(df)
    
    cv = CountVectorizer(max_df=0.5, min_df=100, max_features=100000,stop_words='english')
    vector = cv.fit_transform(df_preprocessed['Processed_Tags']).toarray()
    similarity = cosine_similarity(vector)
    
    return df_preprocessed

if __name__ == "__main__":
    main()