from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
from scipy.io import wavfile
import pandas as pd

import requests
from utils.anime import load_models

import os
from pathlib import Path
from utils.agent import graph
from gtts import gTTS
from langdetect import detect
import assemblyai as aai
import base64
import uuid
import json
import io
from time import sleep

from utils.config import middleware_config

from models.source_code.moviesRecommender import load_recommender, MovieRecommenderSystem # MovieRecommenderSystem is needed for joblib to load the object properly

import environ

env = environ.Env()

environ.Env.read_env(os.path.join(Path(__file__).resolve(), '.env'))

aai.settings.api_key = env('ASSEMBLYAI_API_KEY')


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

def update_tokens():
    global tokens

    data = {
        'client_id': 'd3c72ee839d8f61df73319c576188e48',
        'client_secret': 'f20899b47ab1a30466db5804f57391bcc857690e8ec9c9b76ee7b7661ecb7c57',
        'grant_type': 'refresh_token',
        'refresh_token':tokens['refresh_token']
    }

    response = requests.post('https://myanimelist.net/v1/oauth2/token', data=data)

    if response.status_code == 200:
        with open('tokens.json', 'w') as file:
            json.dump(response.json(), file)
        tokens = response.json()
    
    else:
        return -1

@app.get('/get-anime/{animeID}/')
def get_anime(animeID):
    response = requests.get(f'https://api.jikan.moe/v4/anime/{animeID}')
    
    if response.status_code == 404:
        raise HTTPException(404, 'Not Found')

    if response.status_code == 400:
        raise HTTPException(400, 'Could not get anime data')
    

    if response.status_code == 200:
        anime = response.json()
        sleep(0.5)
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


@app.post('/agent/')
def agent(query: AgentInput):
    data = np.array(json.loads(query.data), dtype=np.float32)
    rate = int(query.rate)
    if query.id is None:
        id_ = str(uuid.uuid4())
    else:
        id_ = query.id

    audio_int16 = (data * 32767).astype(np.int16)
    audio_bytes = io.BytesIO()
    wavfile.write(audio_bytes, rate, audio_int16)
    audio_bytes.seek(0)

    try:
        
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1, language_code='en_us')
        transcript = aai.Transcriber(config=config).transcribe(audio_bytes)

        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        message = transcript.text

        if message == '' or message == ' ':
            raise RuntimeError()

    except RuntimeError:
        return {'nothing': 'nothing'}
    
    output = graph.invoke({
        'messages': [
            {'role': 'system', 'content': '''
                You are a helpful assistant, you can search the web.
                You must follow the rules delimited by backticks.
                The rules: ```
                - Do not use emojis in your responses
                - Respond with the same language the user used to talk to you
                - Make your responses five sentences at most
                - If you are asked to search the web, use web_earch tool, extract the urls from it's output, and format your response as JSON with the following key:
                    urls: <the list of urls extracted from web_search tool output>
                ```
            '''},
            {'role': 'human', 'content': message}
        ]
    }, config={'configurable': {'thread_id': id_}}, stream_mode='values')['messages'][-1].content

    urls = []
    if 'json' in output:
        print(output)
        content = output.split("```")[1].split("json\n")[1][:-1]
        urls = json.loads(content)['urls']
        print(urls)

    lang = detect(output)

    speech = gTTS(text=output, lang=lang)
    stream = speech.stream()
    b = b''.join(stream)
    audio_data = base64.b64encode(b).decode('UTF-8')

    if len(urls) > 0:
        audio_data = ''
        output = ''

    response = {
        'urls': urls,
        'audio_data': audio_data,
        'ai_message': output,
        'id': id_
    }

    return response




if not PRODUCTION:
    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, host='0.0.0.0', port=9000)
