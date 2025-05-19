import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI 
from pydantic import BaseModel 

app = FastAPI()

env_path = Path("../secrets/.env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GROQ_API")

LLM_API_URL = "http://localhost:5000/v1/chat/completions"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    payload = {
        "messages": [
            {
            "role": "user",
            "content": data.prompt 
            }
        ],
        "mode": "instruct",
        "character": "Assistant",
        "max_tokens": 800,
        "top_p": 0.9,
        "temperature": 1,
        "seed": 10,

    }


    response = requests.post(LLM_API_URL, json=payload)
    result = response.json()
    
    return {
        "response": result["choices"][0]["message"]["content"]
    }

# print(result)
