import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import re


# Define TextSelector class
class TextSelector:
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.key in X.columns:
                return X[self.key].fillna('')
            else:
                return ''
        elif isinstance(X, str):
            return X
        else:
            raise ValueError("Invalid input type. Must be DataFrame or string.")


# Load the dataset
data = pd.read_csv("archive/ratings_sample.csv")

# Preprocess text data using CountVectorizer
text_transformer = Pipeline(steps=[
    ('selector', TextSelector(key='overview')),
    ('text_vectorizer', CountVectorizer())
])

# Preprocess genre data using CountVectorizer
genre_transformer = Pipeline(steps=[
    ('selector', TextSelector(key='genres')),
    ('genre_vectorizer', CountVectorizer())
])

# Combine text and genre features
preprocessor = FeatureUnion(transformer_list=[
    ('text_transformer', text_transformer),
    ('genre_transformer', genre_transformer)
])

# Prepare data for model training
X = data[['genres', 'overview']]
y = data['rating']

# Build pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)


# Function to preprocess movie data
def preprocess_movie_data(movie_name, genre):
    cleaned_movie_name = re.sub(r'[^\w\s]', '', movie_name)
    return pd.DataFrame({'genres': [genre], 'overview': [cleaned_movie_name]})


# Function to predict rating for a given movie name and genre
def predict_rating(movie_name, genre):
    movie_data = preprocess_movie_data(movie_name, genre)
    # Predict rating using the model
    rating_prediction = pipeline.predict(movie_data)
    print("Predicted rating for the movie '{}' is: {:.2f}".format(movie_name, rating_prediction[0]))


# Loop until the user inputs 'q' to exit
while True:
    # Take user input for movie name
    user_movie_name = input("Enter the name of the movie (or 'q' to quit): ")

    if user_movie_name.lower() == 'q':
        print("Exiting...")
        break

    # Take user input for movie genre
    user_movie_genre = input("Enter the genre of the movie: ")

    # Predict rating for the user-provided movie information
    predict_rating(user_movie_name, user_movie_genre)
