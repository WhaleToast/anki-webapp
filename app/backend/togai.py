from pathlib import Path
from together import Together
from dotenv import load_dotenv 
import os

envPath = Path("../secrets/.env")
load_dotenv(dotenv_path=envPath)
apiKey = os.getenv("togetherAPI")

client = Together(api_key=apiKey)
res = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[{"role": "user", "content": "Hello"}],
)

print(res.json)
