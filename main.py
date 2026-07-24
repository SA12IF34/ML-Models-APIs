# load Environment Variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests
from utils.anime import load_models

import os
from pathlib import Path
import assemblyai as aai
import json
import os
from pathlib import Path
from time import sleep

from utils.config import middleware_config

from models.source_code.moviesRecommender import load_recommender, MovieRecommenderSystem # MovieRecommenderSystem is needed for joblib to load the object properly
import __main__
__main__.MovieRecommenderSystem = MovieRecommenderSystem



aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')


PRODUCTION = False

class AnimeProfile(BaseModel):
    profile: list[int]

class IMDBProfile(BaseModel):
    seen: list[str]
    ratings: list[float]

class AgentInput(BaseModel):
    data: str
    rate: float | int
    id: str | None = None



app = FastAPI()
app.add_middleware(**middleware_config(PRODUCTION))

tokens = json.load(open('tokens.json'))



@app.get('/get-anime/{animeID}/')
def get_anime(animeID):
    response = requests.get(
        f'https://api.myanimelist.net/v2/anime/{animeID}?fields=id,title,main_picture,,,synopsis,mean,rank,media_type,status,genres', headers={
            'Authorization': f"Bearer {tokens['access_token']}"
        })
    
    if response.status_code == 404:
        raise HTTPException(404, 'Not Found')

    if response.status_code == 400:
        raise HTTPException(400, 'Could not get anime data')


    if response.status_code == 429:
        return HTTPException(429, 'Rate Limited')

    if response.status_code == 200:
        anime = response.json()
        
        return anime

    raise HTTPException(500, 'Internal Server Error')


anime_recommender = load_models()
movie_recommender = load_recommender()

@app.post('/recommend-anime/')
def recommend_anime(profile: AnimeProfile):

    complete_profile = anime_recommender.make_profile(profile.profile)    
    anime_ids = anime_recommender.recommend(complete_profile, profile.profile)

    recommendations = []

    for id_ in anime_ids:
        sleep(0.6)
        response = requests.get(f'https://api.jikan.moe/v4/anime/{id_}')
        if response.status_code == 200:
            recommendations.append(response.json())
        
        else:
            continue


    return {"recommendations": recommendations}


omdb_apikey = env('OMDB_API_KEY')

@app.get('/get-imdb/{imdbID}/')
def get_imdb(imdbID):

    response = requests.get(f'http://www.omdbapi.com/?i={imdbID}&apikey={omdb_apikey}')
    
    data = response.json()

    if data and 'Response' in data and data['Response'] == 'True':
        return data

    
    raise HTTPException(400, 'Could not get imdb material, imdbID may not be valid')


@app.post('/recommend-imdb/')
def recommend_imdb(profile: IMDBProfile):
    complete_profile, _ = movie_recommender.make_profile(profile.seen, profile.ratings)

    recommendation_data = movie_recommender.recommend_movies(complete_profile)
    
    recommendations = []
    for imdbID in recommendation_data['itemId'].values():
        response = requests.get(f'http://www.omdbapi.com/?i={imdbID}&apikey={omdb_apikey}')

        data = response.json()
        if data and 'Response' in data and data['Response'] == 'True':
            recommendations.append(data)

    return {"recommendations": recommendations}




if not PRODUCTION:
    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, host='0.0.0.0', port=9000)
