import numpy as np
import pandas as pd

from typing import Iterable
import nltk
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

import os

class MovieRecommenderSystem:
  """
  A content-based movie/TV series recommender system built on top of imdb dataset that leverages user profiles.
  """

  base_path = ""
  imdb_df = None
  imdb_rating_df = None
  vectorizer = None
  vectorized_genre_labels = None

  def __init__(self, base_path=""):
   
    try:
      nltk.data.find('corpora/stopwords')
      nltk.data.find('tokenizers/punkt')
      nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
      nltk.download('stopwords')
      nltk.download('punkt')
      nltk.download('punkt_tab')
   
    try:
      self.base_path = base_path
      self.imdb_df = pd.read_csv(self.base_path+'imdb.csv')
      self.vectorizer = joblib.load(self.base_path+'vectorizer.joblib')
      self.vectorized_genre_labels = self.vectorizer.get_feature_names_out()
    
    except FileNotFoundError:
      print('***There is no predefined data, you must use `set_new_data` method to set the data will be used.***')

    except:
      print('***An error occured, make sure you have data or load new data with `set_new_data`, and check source code of the object.***')
  
  def set_new_data(self, data_filepath: str, rating_filepath: str) -> None:
    """
    Preprocesses the new data, and saves it to the `base_path`.
    """
    print("Preparing data for preprocessing...")

    self.imdb_df = pd.read_csv(data_filepath, sep='\t')
    self.imdb_rating_df = pd.read_csv(rating_filepath, sep='\t')

    self.imdb_rating_df = self.imdb_rating_df[self.imdb_rating_df['averageRating'] >= 7]

    current_year = self.get_current_year()

    start_year = self.imdb_df['startYear'].replace(r'\N', np.nan)

    start_year = start_year.astype(np.float64)
    start_year = start_year[(start_year <= current_year) & (start_year >= 1980)]

    start_year = start_year.astype(int).astype(str)

    self.imdb_df = self.imdb_df[(self.imdb_df['titleType'].isin(['movie','tvMovie', 'tvSeries'])) & (self.imdb_df['startYear'].isin(start_year))]
    self.imdb_df.drop(['originalTitle', 'isAdult', 'endYear', 'runtimeMinutes'], axis=1, inplace=True)

    self.imdb_df = self.imdb_df[self.imdb_df['tconst'].isin(self.imdb_rating_df['tconst'].unique())]
    self.imdb_df = pd.merge(self.imdb_df, self.imdb_rating_df[['tconst', 'averageRating']], on='tconst', how='left')
    self.imdb_df = self.imdb_df.rename(columns={'averageRating': 'rating'})

    print('Finished preparing data...')

    self._preprocess_data()

  def set_base_path(self, base_path):
    self.base_path = base_path
    return

  def _preprocess_data(self):

    print('Start preprocessing data...')

    self.vectorizer = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b[a-zA-Z-]+\b')
    vectorized_genres = self.vectorizer.fit_transform(self.imdb_df['genres'])

    self.vectorized_genre_labels = self.vectorizer.get_feature_names_out()

    joblib.dump(self.vectorizer, self.base_path+'vectorizer.joblib')

    genres_df = pd.DataFrame(vectorized_genres.toarray(), columns=self.vectorized_genre_labels)
    self.imdb_df = pd.merge(self.imdb_df, genres_df, left_index=True, right_index=True)


    self.imdb_df.to_csv(self.base_path+'imdb.csv', index=False)
    self.imdb_rating_df.to_csv(self.base_path+'rating.csv', index=False)

    vectorized_genres = None
    genres_df = None

    print('Finished preprocess data!')

  def get_current_year(self) -> int:

    return datetime.now().year


  def get_genre_labels(self) -> np.ndarray:
    assert self.vectorizer is not None
    return self.vectorizer.get_feature_names_out()


  def make_profile(self, seen_items: Iterable, ratings:Iterable, recommend=False):
    assert self.imdb_df is not None
    assert len(seen_items) > 0 and len(ratings) > 0
    assert len(seen_items) == len(ratings)


    ratings_sum = np.zeros(len(self.vectorized_genre_labels))
    genres_sum = np.zeros(len(self.vectorized_genre_labels))

    i = 0
    for item in seen_items:
      values = self.imdb_df[self.imdb_df['tconst'] == item][self.vectorized_genre_labels].values
      if len(values) == 0:
        i+=1
        continue

      values = np.float64(values[0])

      genres_sum += values
      ratings_sum += np.float64(values*ratings[i])
      i+=1

    profile = np.nan_to_num(np.float64(ratings_sum / genres_sum), nan=0.0)

    if recommend:
      return profile, self.recommend_movies(profile)

    return profile, None

  def recommend_movies(self, profile: np.ndarray, rcmd_size: int=20):

    assert type(profile) == np.ndarray
    assert profile.ndim == 1
    assert profile.shape == self.vectorized_genre_labels.shape

    imdb_vector = self.imdb_df[self.vectorized_genre_labels].values
    imdb_vector = imdb_vector

    recommendation_scores = np.dot(imdb_vector, profile)
    recommendations_df = pd.DataFrame({
      'itemId': self.imdb_df['tconst'],
      'startYear': self.imdb_df['startYear'],
      'score': recommendation_scores
    })

    recommendations_df = recommendations_df.sort_values(by=['score', 'startYear'], ascending=False).head(rcmd_size)
    
    return recommendations_df.to_dict()

def load_recommender():
  movie_recommender = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'movie_recommendation')+'\\recommender_model.joblib')

  return movie_recommender

