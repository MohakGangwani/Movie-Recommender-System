# Movie Recommendation System

This repository contains a **Movie Recommendation System** project developed using Python and Flask. It leverages machine learning techniques to recommend movies based on user preferences and historical data.

## Features

- **Content-Based Filtering**: Recommends movies based on their metadata, such as genres, cast, crew, and descriptions.
- **Interactive Web Interface**: Built using Flask, allowing users to search for movies and view personalized recommendations.

## Machine Learning Workflow

1. **Data Preprocessing**:
   - Utilized the TMDB dataset from Kaggle, which includes rich metadata on movies.
   - Cleaned and preprocessed data to handle missing values, duplicates, and inconsistencies.

2. **Feature Engineering**:
   - Extracted relevant features such as genres, keywords, and cast information.
   - Employed natural language processing (NLP) techniques to process textual data.

3. **Similarity Calculation**:
   - Implemented K-Nearest Neighbors (KNN) and TF-IDF vectorization to measure the closeness of movie features.
   - Created a similarity matrix to efficiently retrieve recommendations.

4. **Model Deployment**:
   - Integrated the recommendation logic into a Flask web application.
   - Ensured scalable and efficient query handling for real-time recommendations.

## Tools and Libraries Used

- **Programming Language**: Python
- **Web Framework**: Flask
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (KNN, TF-IDF Vectorizer)
- **Natural Language Processing**: Scikit-learn (TF-IDF Vectorizer)

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/MohakGangwani/Movie-Recommender-System.git
   ```

2. Navigate to the project directory:
   ```bash
   cd movie-recommendation-system
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to use the application.

## Dataset

The project uses the **TMDB dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/).

Ensure the dataset is placed in the appropriate directory (`/data/`) before running the application.

## Future Enhancements

- Integrate a **hybrid recommendation system** combining content-based and collaborative filtering.
- Implement advanced techniques like **Matrix Factorization** or **Deep Learning** for improved recommendations.
- Deploy the application on cloud platforms like AWS or Heroku for wider accessibility.

## Contribution

Contributions are welcome! If you have ideas to improve the project or want to fix any issues, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request detailing your changes.

---

### Contact

If you have any questions or feedback, feel free to reach out:

- **Name**: Mohak Gangwani
- **Email**: mohakmg99@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/mohak-gangwani/

---
Thank you for checking out the Movie Recommendation System project!
