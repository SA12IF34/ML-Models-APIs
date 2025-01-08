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


PRODUCTION = False

class KMeansProfileInput(BaseModel):
    profile: list[str]
    seen_animes: list[int]

class AgentInput(BaseModel):
    message: str
    history: list


app = FastAPI()
app.add_middleware(**middleware_config(PRODUCTION))

tokens = {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjBhMTY0MjNhMzFjOWJkOWE5M2QwYmVlMzI0ZmMyMTE0ZmMwZGU5MGE4MDk5NWE4NDA2NDM4NjhiN2YyNGFmMjJkYzhmMzgyMTE3OTdiNmE3In0.eyJhdWQiOiJiYzdhYmIzOTNjZjRmYmIwZWVmYTMyNTI5ZDk0ZDg5MCIsImp0aSI6IjBhMTY0MjNhMzFjOWJkOWE5M2QwYmVlMzI0ZmMyMTE0ZmMwZGU5MGE4MDk5NWE4NDA2NDM4NjhiN2YyNGFmMjJkYzhmMzgyMTE3OTdiNmE3IiwiaWF0IjoxNzM2MzY1NjQ4LCJuYmYiOjE3MzYzNjU2NDgsImV4cCI6MTczOTA0NDA0OCwic3ViIjoiMTc2OTkwMjUiLCJzY29wZXMiOltdfQ.Em_RzMf1DKZ-sf0ph_PSM8kJBaynbJEqDHaOvDdMdAc9d9yWuot6eWJd21XM9coYjMeeqeTOZ9ekx0NyC226Xg9tW71YLCrAIPCEFqFPG8SoN6PehukTX1nsdBe5hlTJ9umXLeh1R1Q7KfyUARIH6VYyjLpXb3xM_b82NX0Xpdh5NojrNYrnc22qQm1zpCApV1c20saEG1pupPjzjOLEfODfxx4k0xL_of6dKfiV2csnk6zQPoxbRhCi93KVwCVLXXf5kopb1NJDjvhLOyfhzAH-qkdpW1NfiJpBKT7LYgeJ6JE3RkvzmmqGSw_M0IczF9nc3F-eR86ASk1SFHJaMw",
    "refresh_token": "def502005f51d7d66c4bf4fc5f4698968da9381017dac98d5240cfe20ed6bbb4b049279fb9fff3937c28b39137eed5ea748ae6a37aa9e1fd4a575d12fb6990ea4d7698438c26f382e7a6289115082c7a35b5465db332b7e1806c7a2f67e0fee843c4f2df61548637d7284bb6af1512d98ac9986ad2f12be48f17f0e182871e45adf80a825439dc54e49082e0ed687be305bb5c6890df7cf98895d87988443893a1a0144c130e642ae61bc88e250f03bd9da6be8a5702c8a5f813d179d2ed28f71119ee2b84e877364efecb8202f4b48ced1858696f5e2d348d6bd50318f02baad7e381fa9eebcd061ca0cd57f97bcc2ad47838b2fb368628666c12e5ede04f8475075b18b90e61bbb7ac5554a93b4f146c8b0436073b386405a1d38d8e54b51eac1acfd20598cdb4afe7bdaaff0b0532ab54cec0cafe840788a7537e25afc80cd60f8fb0cf6ea77d65ae3b70b9890dcccffee87bffce180a92854aa4cfdd32093a66229c96958e1c5466a5caff16521e2fd238e91b3bf58618647fe3230de2e0f6ad975d7700f7b830"
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
