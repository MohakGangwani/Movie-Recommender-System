from flask import Flask, render_template, request
import math
from src.utils import *

app = Flask(__name__)

# Preload data
params = load_params()
print(params)
MOVIES_DF = load_preprocess_data(params).reset_index().rename(columns={'index': 'id'})
MOVIES_DF['processed_title'] = MOVIES_DF['title'].str.lower()
MOVIES_LIST = MOVIES_DF.to_dict(orient='records')
MOVIES_PER_PAGE = 25
# Preprocess MOVIES_LIST into a dictionary for quick lookup
MOVIES_DICT = {movie['id']: movie for movie in MOVIES_LIST}

# Preload recommendation system data
vectors = get_vector(None, params)  # Assuming this loads all vectors at once
model = get_sim_model(vectors, params)  # Load the model only once
PRELOADED_DATA = {
    "movies": MOVIES_DF.set_index("id").to_dict(orient="index"),  # Dictionary for quick ID lookups
    "vectors": vectors,
    "model": model,
    "params": params,
}


@app.route('/')
def index():
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))

    # Filter and paginate movies
    filtered_movies = [movie for movie in MOVIES_LIST if query in movie['processed_title']] if query else MOVIES_LIST
    total_movies = len(filtered_movies)
    total_pages = math.ceil(total_movies / MOVIES_PER_PAGE)
    start_idx = (page - 1) * MOVIES_PER_PAGE
    movies_to_display = filtered_movies[start_idx:start_idx + MOVIES_PER_PAGE]

    return render_template(
        'index.html',
        movies={movie['id']: movie for movie in movies_to_display},
        query=query,
        page=page,
        total_pages=total_pages
    )



@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # Find movie by ID
    movie = MOVIES_DICT.get(movie_id)
    if not movie:
        return "Movie not found", 404

    # Get recommendations using preloaded data
    recommendations = recommend_movie(movie_id, PRELOADED_DATA)
    return render_template('movie.html', movie=movie, recommendations=recommendations)



if __name__ == '__main__':
    app.run(debug=True)