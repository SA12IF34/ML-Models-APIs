# load Environment Variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests
from utils.anime import load_models, API_QUERY

import os
import json
import os

from utils.config import middleware_config

from models.source_code.moviesRecommender import load_recommender, MovieRecommenderSystem # MovieRecommenderSystem is needed for joblib to load the object properly
import __main__
__main__.MovieRecommenderSystem = MovieRecommenderSystem


PRODUCTION = False

# Schemas
class AnimeProfile(BaseModel):
    profile: list[int]

class IMDBProfile(BaseModel):
    seen: list[str]
    ratings: list[float]

class AgentInput(BaseModel):
    data: str
    rate: float | int
    id: str | None = None

class AnimeID(BaseModel):
    id_list: list[int]

anime_recommender = None
movie_recommender = None

def lifespan(app: FastAPI):
    global anime_recommender, movie_recommender
    anime_recommender = load_models()
    movie_recommender = load_recommender()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(**middleware_config(PRODUCTION))


@app.post('/get-anime')
def get_anime(anime: AnimeID):
    
    variables = {
        'idMal': anime.id_list
    }
    api_link = 'https://graphql.anilist.co'
    response = requests.post(api_link, json={'query': API_QUERY, 'variables': variables})
    print(response.status_code)
    return response.json()


@app.post('/recommend-anime/')
def recommend_anime(profile: AnimeProfile):

    complete_profile = anime_recommender.make_profile(profile.profile)    
    anime_ids = anime_recommender.recommend(complete_profile, profile.profile)

    recommendations = []
    response = requests.post('https://graphql.anilist.co', json={'query': API_QUERY, 'variables': {'idMal': anime_ids}})
    recommendations.extend(response.json()['data']['Page']['media'])


    return {"recommendations": recommendations}


omdb_apikey = os.getenv('OMDB_API_KEY')
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
