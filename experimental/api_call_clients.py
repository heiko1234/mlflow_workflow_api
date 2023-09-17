



import os
from io import StringIO
from json import dumps, loads
from os import getenv

import requests
from dotenv import load_dotenv
from flask import request


from werkzeug.datastructures import Headers, EnvironHeaders




class BackendClient:
    def __init__(self, base_url):
        self._base_url = base_url
        
    def _check_endpoint(self, endpoint):
        
        if endpoint[0] == "/":
            output = endpoint[1:]
        
        else:
            output = endpoint
        
        return output

    def _get_headers(self):
        try:
            auth_header = request.headers
            output = auth_header
        
        except Exception as e:
            print(f"headers exception: {e}")
            output = {}
            
        return output
    
    def execute_get(self, endpoint: str, headers=None, params: dict=None, data=None):
        
        endpoint = self._check_endpoint(endpoint=endpoint)
        
        if headers is None:
            output = requests.get(
                url=self._base_url + "/" + endpoint,
                headers=self._get_headers(),
                params=params,
                data=data
            )
        else:
            output = requests.get(
                url=self._base_url + "/" + endpoint,
                headers=headers,
                params=params,
                data=data
            )
        return output
    
    def execute_post(self, endpoint: str, headers=None, params: dict=None, data=None, json=None):
        
        endpoint = self._check_endpoint(endpoint=endpoint)
        
        if headers is None:
            output = requests.post(
                url=self._base_url + "/" + endpoint,
                headers=self._get_headers(),
                params=params,
                data=data,
                json=json
            )
        else:
            output = requests.post(
                url=self._base_url + "/" + endpoint,
                headers=headers,
                params=params,
                data=data,
                json=json
            )
        return output



class APIBackendClient:
    def __init__(self):
        load_dotenv()
        local_run = getenv("LOCAL_RUN", False)
        
        if local_run:
            base_url = "http://127.0.0.1:8000"
            
            print(f"use local base_url for backend service connect via eg. port forwarding: {base_url}")
            
        else:
            base_url = "http://backend:8000"     # any backend service
            
            print(f"use docker base_url for backend service connect via docker network: {base_url}")
            
        self.Backendclient = BackendClient(base_url=base_url)
























