from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import pandas as pd

import requests
from utils.anime import load_data, load_models, get_recommendations

from utils.agent import agent_executor
from langchain_core.messages import HumanMessage, AIMessage
from gtts import gTTS
import base64

from utils.config import middleware_config


PRODUCTION = True

class KMeansProfileInput(BaseModel):
    profile: list[str]
    seen_animes: list[int]

class AgentInput(BaseModel):
    message: str
    history: list


app = FastAPI()
app.add_middleware(**middleware_config(PRODUCTION))

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

    # raise HTTPException(500, 'Internal Server Error')


anime_clusters = load_data(PRODUCTION)
vectorizer, scaler, recommender = load_models()

@app.post('/recommend-anime/')
def recommend_anime(profile: KMeansProfileInput):

    profile_array = np.array(profile.profile)
    profile_array = (90-len(profile.seen_animes)) * 8.5 * np.sum(vectorizer.transform(profile_array).todense(), axis=0)
    profile_array = scaler.transform(np.array(profile_array))

    
    recommendations_df = get_recommendations(anime_clusters, recommender, profile_array)
    anime_ids = recommendations_df['anime_id'].values

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


@app.post('/agent/')
def agent(query: AgentInput):
    history = query.history
    message = query.message

    history.append(HumanMessage(content=message))

    output = agent_executor.invoke({'input': message, 'chat_history': history})['output']
    history.append(AIMessage(content=output))
    urls = []
    if 'search-results' in output:
        if '``' in output:
            urls = output.split('``')[1].split(',')
        else:
            urls = output[:-1].split('search-results:')[1].split(',')

    speech = gTTS(output)
    stream = speech.stream()
    b = b''.join(stream)
    audio_data = base64.b64encode(b).decode('UTF-8')

    if len(urls) > 0:
        audio_data = ''
        output = ''

    response = {
        'urls': urls,
        'history': history,
        'audio_data': audio_data,
        'ai_message': output
    }

    return response




if not PRODUCTION:
    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, host='0.0.0.0', port=9000)
