from flask import Flask, render_template, request, url_for
import math
from src.utils import *

app = Flask(__name__)

# Sample data for demonstration
params = load_params()
MOVIES = load_preprocess_data(params).reset_index().rename(columns={'index':'id'})
MOVIES['processed_title'] = MOVIES['title'].str.lower()
MOVIES_PER_PAGE = 25


@app.route('/')
def index():
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))

    filtered_movies = MOVIES.loc[MOVIES["processed_title"].str.contains(query)].T.to_dict()
    total_movies = len(filtered_movies)
    total_pages = math.ceil(total_movies / MOVIES_PER_PAGE)
    
    idx = list(filtered_movies.keys())[:25]
    movies_to_display = {k:filtered_movies[k] for k in idx}
    
    return render_template('index.html', movies=movies_to_display, query=query, page=page, total_pages=total_pages)


@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = MOVIES.T.to_dict()[movie_id]
    
    if not movie:
        return "Movie not found", 404
    
    recommendations = recommend_movie(movie_id, params)
    return render_template('movie.html', movie=movie, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)