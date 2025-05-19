import requests
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("../secrets/.env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GROQ_API")
print(api_key)

