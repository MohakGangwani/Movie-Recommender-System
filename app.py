from flask import Flask, render_template, request, url_for
import math
from src.utils import *
import json

app = Flask(__name__)

# Sample data for demonstration
params = load_params()
MOVIES = load_preprocess_data(params).dropna().reset_index().rename(columns={'index':'id'})
MOVIES['processed_title'] = MOVIES['title'].str.lower()
MOVIES_PER_PAGE = 25


@app.route('/')
def index():
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))

    filtered_movies = MOVIES.loc[MOVIES["processed_title"].str.contains(query)].T.to_dict()
    total_movies = len(filtered_movies)
    total_pages = math.ceil(total_movies / MOVIES_PER_PAGE)
    
    start_idx = (page - 1) * MOVIES_PER_PAGE
    end_idx = start_idx + MOVIES_PER_PAGE
    movies_to_display = {k:filtered_movies[k] for k in range(start_idx, end_idx)}
    print(type(movies_to_display))
    print(movies_to_display)

    return render_template('index.html', movies=movies_to_display, query=query, page=page, total_pages=total_pages)


@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = next((m for m in MOVIES if m["id"] == movie_id), None)
    if not movie:
        return "Movie not found", 404

    recommendations = [m for m in MOVIES if m["id"] in movie["recommendations"]]
    return render_template('movie.html', movie=movie, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)