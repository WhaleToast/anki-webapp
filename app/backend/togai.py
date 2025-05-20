import os
from pathlib import Path
from together import Together
from dotenv import load_dotenv 
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware

envPath = Path("../../secrets/.env")
load_dotenv(dotenv_path=envPath)
apiKey = os.getenv("togetherAPI")




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    client = Together(api_key=apiKey)
    res = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": data.prompt}],
        top_p=0.95,
        temperature=1.2,

    )

    return {
        "response": res.choices[0].message.content
    }
