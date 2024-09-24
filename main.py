from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

import requests

PRODUCTION = True

class KMeansProfileInput(BaseModel):
    profile: list[str]
    seen_animes: list[int]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'https://saifchan.online',
        'https://cms.saifchan.online',
        'https://ml-models.saifchan.online',
        'http://127.0.0.1:8000',
        'http://localhost:4173'
    ],
    allow_credentials=True,
    allow_headers=['*'],
    allow_methods=['*']
)

tokens = {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjZmNzhhNmZmMWU4OWFhMjcwZWVlN2Y4NWNmZjNkMzE1ZTZhNTMzNjYwY2JlM2YyNmFkMzkyYmI1NzM4MzA0M2I2OGE4NmRhMDc2OTA0OGEyIn0.eyJhdWQiOiJiYzdhYmIzOTNjZjRmYmIwZWVmYTMyNTI5ZDk0ZDg5MCIsImp0aSI6IjZmNzhhNmZmMWU4OWFhMjcwZWVlN2Y4NWNmZjNkMzE1ZTZhNTMzNjYwY2JlM2YyNmFkMzkyYmI1NzM4MzA0M2I2OGE4NmRhMDc2OTA0OGEyIiwiaWF0IjoxNzI2NjQyOTE2LCJuYmYiOjE3MjY2NDI5MTYsImV4cCI6MTcyOTIzNDkxNiwic3ViIjoiMTc2OTkwMjUiLCJzY29wZXMiOltdfQ.rvWDfBoL9nq07pyeicQx4UfU7JB-eGkOQESRWhmBI-cXY8OZ7IrD3Jxmv-vsK3NVBNMHD9baX4IjL6rpQSl9CxIjq0QrC31qXOL7o8qd5pwgIg0-IV3W0yHtOTEkrC0CpksChWDwABIXNrU7HdFF7gYp0MQ3B2rYzxL_5Mcq4HyXvahr1XtXWNJA9VjvhI1oZIvDAr63UECSsSmoYb88KQKuH7HzXrmAiCKeaKS4xVwYmJXBoc2MLwO7764-Axq7WDX-OV7zogO0uD73bj6g0jAKMayfUcC-2dBpCNDLNbBud8dSrE6wJHUowdpiBzQEBXeFdCwBR3Hr8gdaEv_VRA",
    "refresh_token": "def50200321bceaac27dd88660ea90673932b6977eaa2a0d6af43cb4d1cf25439b5c48ad0cf368ba239bdd486050dc8a2ec1aa3469a43f671a589d11409a601643e000734b2e6896e8f4c7b4ad341f433eec543bde50dfe73e24b8b2f41ca9d92fc32e889eb2a8f7802b8e0ebff92aaf143815f037ccc0a4124145c64c7ab4b4e5fb5a545ef21e2fb212fe16f5165ef049d954cada22eee916261fcd4cb5b348a3d96146eefc7e6c60e34eab5b9ee247939df0a71ee3af587621cd5bf0ada2fe753ed4eb33c3a025ad28f5ae097020d575ef87df67973731e4660e4df9c97d40a4d0946e51dfca8ac0dac37cb0e77aa02c332cb030bcf4944ab32d040f6a46de3ad1899f9f17174f6c0cbb5c1853565b936fd7aa749d0145fa91a426466cb6c88c07b6ea8fc79e8d3bca1c90458cfa1d29f0ce0529d2b514310c140ced18a0e6ec0a7d34ccd915f17c0a7f30e404c137455a353b0f20d02016a519b68a733d510a986b3cd3231f485d94e8830f9025359b742c0b04f7a803d75a0f3e14f7423f31249b0868af2c2096"
}

@app.get('/get-anime/{animeID}/')
def get_anime(animeID):
    response = requests.get(f'https://api.myanimelist.net/v2/anime/{animeID}?fields=id,title,synopsis,main_picture,mean,genres', headers={
        'Authorization': 'Bearer '+tokens['access_token']
    })
    if response.status_code == 404:
        raise HTTPException(404, 'Not Found')

    if response.status_code == 200:
        anime = response.json()

        return anime

    raise HTTPException(500, 'Internal Server Error')


if PRODUCTION:
    anime_clusters = load_dataset('csv', data_files='hf://datasets/SaifChan/AnimeDS/full_cluster_new.csv')['train'].to_pandas()
    anime_df = load_dataset('csv', data_files='hf://datasets/SaifChan/AnimeDS/anime.csv')['train'].to_pandas()
    
    # anime_clusters = pd.DataFrame(dataset1['train'])
    # anime_df = pd.DataFrame(dataset2['train'])
else:
    anime_clusters = pd.read_csv('data/full_cluster_new.csv')
    anime_df = pd.read_csv('data/anime.csv')

anime_df = anime_df[~anime_df['genre'].str.contains('hentai', case=False, na=False)]
anime_df.dropna(inplace=True)
cluster_counts = anime_clusters.groupby(['anime_id'])['cluster'].nunique().reset_index()
cluster_counts.columns = ['anime_id', 'cluster_count']
anime_clusters = pd.merge(anime_clusters, cluster_counts, on='anime_id')
del cluster_counts

@app.post('/recommend-anime/')
def recommend_anime(profile: KMeansProfileInput):
    profile_array = np.array(profile.profile)

    vectorizer = joblib.load('./models/vectorizer.joblib')
    scaler = joblib.load('./models/anime_scaler_new.joblib')
    kmeans_model = joblib.load('./models/anime_recommender_new.joblib')
    
    profile_array = vectorizer.transform(profile_array).todense()
    profile_array = (90-len(profile.seen_animes)) * 6.6 * np.sum(profile_array, axis=0)
    profile_array = scaler.transform(np.array(profile_array))

    cluster_label = kmeans_model.predict(profile_array)[0]
    unseen_animes = anime_clusters[~anime_clusters['anime_id'].isin(profile.seen_animes)]
    unseen_animes  = unseen_animes[unseen_animes['anime_id'].isin(anime_df['anime_id'].values)]
    anime_ids = unseen_animes[unseen_animes['cluster'] == cluster_label].drop_duplicates(subset=['anime_id']).sort_values(['rating', 'cluster_count'], ascending=[False, True]).head(16)['anime_id'].values.tolist()
    # anime_ids = anime_ids.sample(16, random_state=42)['anime_id'].values.tolist()
    recommendations = []

    for id_ in anime_ids:
        response = requests.get(f'https://api.myanimelist.net/v2/anime/{id_}?fields=id,title,synopsis,main_picture,mean,genres', headers={
        'Authorization': 'Bearer '+tokens['access_token']
        })
        if response.status_code == 200:
            recommendations.append(response.json())
        
        else:
            continue


    return {"recommendations": recommendations}


if not PRODUCTION:
    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, host='0.0.0.0', port=9000)
