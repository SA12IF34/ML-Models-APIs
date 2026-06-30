from fastapi.middleware.cors import CORSMiddleware

def middleware_config(production:bool =False) -> dict:
    '''
    Returns middleware config for the app.
    '''

    if production:
        origins = [
            'https://saifchan.site',
            'https://cms.saifchan.site',
            'https://ml-models.saifchan.site',
            'https://api.ml.saifchan.site'
        ]
    else:
        origins = [
            'https://saifchan.site',
            'https://cms.saifchan.site',
            'https://ml-models.saifchan.site',
            'https://api.ml.saifchan.site',
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