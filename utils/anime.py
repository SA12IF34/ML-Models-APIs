import os
import requests
import json
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

def get_tokens():
    data = {
        'client_id': 'c8267cc23618a74596e37ab933024190',
        'client_secret': '7b970109cdff8df7f35043168f1e16d6019867df96ecfb7dfd48eb59a7f74177',
        'code': 'def50200603e4596fba1b468320f327de06175a3a48da9ed12b62b73e551a54381d2c3a7bc164e87525e1c6c3dcb9b73ec7d30ef058dab8bbdddb2910d406ebac580b5c960b09bd0942ac1c57c489a2b7a4fee4519113258c171fc7442be7412ba503111acba0340c0a8a9f10353f829d406dd7af38c6464580d5f19bba7e7120f3c37a5cb898a29be345e2fec4c03c4d9a6a1959773bb9428136e86991bd568b0d6323c25aa2051917a14501fa54302d327f7eed1a017260b3c773751b78c50c83ae7add930d04571eac88ac8dd187e2fae2c8737826b556f350f8b3fb0f5c6d5040acc95e208b47bad205d948757f12086af544deca8264b006f877c02c7e4fb02405ea0b298e8a50a1116c16594c86a6c95ed8db88ff49b7028e97d9b94bfa7bed36355fcf290434517f9f517e25d69628531fbe5db47e1a591f24291e5f716deeb09dc9e37ca28a2269a9de0db8359ebaa6a79cd5f44f0bdf6ac8c4229dda7509e1c01960d7c653351335f9e0947e48a6346c2e65341ff6c69364d6f405c43cdcf28379e7f46be03ac6fa54b9679b51758c5912693874c7c58f7ed55',
        'grant_type': 'authorization_code',
        'code_verifier': 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz'
    }

    response = requests.post('https://myanimelist.net/v1/oauth2/token', data, headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })
    print(response.status_code)
    print(response.json())
    return response.json()

def update_tokens():
    global tokens

    data = {
        'client_id':'c8267cc23618a74596e37ab933024190',
        'client_secret': '7b970109cdff8df7f35043168f1e16d6019867df96ecfb7dfd48eb59a7f74177',
        'grant_type': 'refresh_token',
        'refresh_token':'def502005b0a3fb4aabfc5cd8c6606e8f10c2a8bc089ef9c035071e74d3eb940fac30ac144a84454fcb66f42aef74044f32afe98acc27d762c38d55af516f6682246b8fd6d65a4b9bec4dcfcd974cc93cf53f7e04aed36814258589ee389183f76727e900692cb0c362d6045dff410bb23eda3d461709526f8b801850ff4a7770c76b7ef77a2ecbad025fc17524d4cac24074b19743b4522a61e77986864d4902732f42b73b6be03c7446d828cfa40f08fff6f1b0ae47017f895270217a8233a8326a491c55219006b1ec6506a2f1de382fa39c79f09799217042362430e4f7413c641d55d8de2f703180b1e09710cc49ae5925e92afe403076fcf24e55c92130bc22db6245adf1105c0d4066142440f94054a07841c7d587a3cf15b8173996f5c9377b89ec0bdb0f7fd965c61f75e7ce8dcd234c1d938c5c28929e71c08a6048864bd88103bb74bb2568eaaadf61796ece8a399a1570829526f4aeff2414bc5ac6a4e3a3f310d9de1c2db5bd9102afb974f955a37c074cd303d96712eaf450fc5074ac6becef6dbad'
    }

    response = requests.post('https://myanimelist.net/v1/oauth2/token', data=data)

    if response.status_code == 200:
        with open('tokens.json', 'w') as file:
            json.dump(response.json(), file)
        tokens = response.json()
    
    else:
        return -1

def get_anime_details(anime_id):
    pass