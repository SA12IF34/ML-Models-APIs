import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset



def load_data(production: bool =False) -> tuple:
    '''
    Loads data and processes it.
    '''
    if production:
        anime_clusters = load_dataset('csv', data_files='hf://datasets/SaifChan/AnimeDS/mean_shift_anime_cluster_df.csv')['train'].to_pandas()
        # anime_df = load_dataset('csv', data_files='hf://datasets/SaifChan/AnimeDS/anime.csv')['train'].to_pandas()

    else:
        anime_clusters = pd.read_csv('data/mean_shift_anime_cluster_df.csv')
        # anime_df = pd.read_csv('data/anime.csv')

    # anime_df = anime_df[~anime_df['genre'].str.contains('hentai', case=False, na=False)]
    # anime_df.dropna(inplace=True)
    # cluster_counts = anime_clusters.groupby(['anime_id'])['cluster'].nunique().reset_index()
    # cluster_counts.columns = ['anime_id', 'cluster_count']
    # anime_clusters = pd.merge(anime_clusters, cluster_counts, on='anime_id')
    # del cluster_counts

    return anime_clusters


def load_models() -> tuple:
    '''
    Loads recommendation and preprocessing models.
    '''

    vectorizer = joblib.load('models/vectorizer.joblib')
    scaler = joblib.load('models/profiles_scaler.joblib')
    kmeans_model = joblib.load('models/mean_shift.joblib')

    return vectorizer, scaler, kmeans_model


def get_recommendations(anime_cluster_df, model, profile):
    label = model.predict(profile)[0]
    recommendations = anime_cluster_df[anime_cluster_df['cluster'] == label].sort_values('rating', ascending=False).head(16)


    return recommendations