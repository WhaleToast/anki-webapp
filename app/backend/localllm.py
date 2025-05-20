# import requests
# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from fastapi import FastAPI 
# from pydantic import BaseModel 
#
# from fastapi.middleware.cors import CORSMiddleware
#
#
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can restrict this later
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# env_path = Path("../secrets/.env")
# load_dotenv(dotenv_path=env_path)
# api_key = os.getenv("togetherAPI")
#
# LLM_API_URL = "http://localhost:5000/v1/chat/completions"
#
# class PromptRequest(BaseModel):
#     prompt: str
#
# @app.post("/generate")
# def generate_anki_cards(data: PromptRequest):
#     payload = {
#         "messages": [
#             {
#             "role": "user",
#             "content": data.prompt 
#             }
#         ],
#         "mode": "instruct",
#         "character": "Assistant",
#         "max_tokens": 4092,
#         "top_p": 0.95,
#         "temperature": 1.2,
#         "seed": -1,
#
#     }
#
#
#     response = requests.post(LLM_API_URL, json=payload)
#     result = response.json()
#     
#     return {
#         "response": result["choices"][0]["message"]["content"]
#     }
#
# # print(result)
