import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset

from models.source_code.animeRecommender import AnimeRecommender


anime_features = ['action', 'adult-cast', 'adventure', 'anthropomorphic',
       'avant-garde', 'award-winning', 'boys-love', 'cars', 'cgdct',
       'comedy', 'dementia', 'demons', 'detective', 'drama', 'ecchi',
       'erotica', 'fantasy', 'female', 'gag-humor', 'game', 'girls-love',
       'gore', 'gourmet', 'harem', 'hentai', 'historical', 'horror',
       'idols', 'isekai', 'iyashikei', 'josei', 'kids', 'love-polygon',
       'magic', 'mahou-shoujo', 'male', 'martial-arts', 'mecha',
       'military', 'music', 'mystery', 'mythology', 'organized-crime',
       'otaku-culture', 'parody', 'performing-arts', 'pets', 'police',
       'psychological', 'racing', 'reincarnation', 'reverse-harem',
       'romance', 'samurai', 'school', 'sci-fi', 'seinen', 'shoujo',
       'shoujo-ai', 'shounen', 'shounen-ai', 'slice-of-life', 'space',
       'sports', 'strategy-game', 'super-power', 'supernatural',
       'survival', 'suspense', 'team-sports', 'thriller', 'time-travel',
       'urban-fantasy', 'vampire', 'video-game', 'workplace', 'yaoi',
       'yuri']


def load_models() -> AnimeRecommender:
    '''
    Loads recommendation and preprocessing models.
    '''

    print('Loading models...')

    anime_recommender = AnimeRecommender(
        'models/anime_recommendation/cluster_model.joblib',
        'models/anime_recommendation/scaler.joblib',
        'models/anime_recommendation/vectorizer.joblib',
        'data/anime/anime_views_per_label.csv',
        'data/anime/processed_anime.csv'
    )

    print('Models loaded successfully.')

    return anime_recommender

