from fastapi.middleware.cors import CORSMiddleware

def middleware_config(production:bool =False) -> dict:
    '''
    Returns middleware config for the app.
    '''

    if production:
        origins = [
            'https://saifchan.online',
            'https://cms.saifchan.online',
            'https://ml-models.saifchan.online'
        ]
    else:
        origins = [
            'https://saifchan.online',
            'https://cms.saifchan.online',
            'https://ml-models.saifchan.online',
            'http://127.0.0.1:8000',
            'http://localhost:4173',
            'http://localhost:5173'
        ]

    config = {
        'middleware_class': CORSMiddleware,
        'allow_origins': origins,
        'allow_credentials': True,
        'allow_headers': ['*'],
        'allow_methods': ['*']
    }

    return config