import os
import asyncio
import time
import re
import tiktoken
import requests
from pathlib import Path
from together import Together
from dotenv import load_dotenv 
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber 
from fastapi import File, UploadFile
from datetime import datetime

envPath = Path("../../secrets/.env")
load_dotenv(dotenv_path=envPath)
apiKey = os.getenv("chutes")
chuteModel = "deepseek-ai/DeepSeek-V3-0324"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)
def split_paragraphs(text:str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}|\n(?=\S)', text) if p.strip()]

def token_safe_chunking(text: str, max_tokens: int = 6500, model: str = "gpt-3.5-turbo") -> list[str]:
    enc = tiktoken.encoding_for_model(model)
    paragraphs = split_paragraphs(text)    

    chunks = []
    current_chunk = []
    current_tokens = 0 

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        if current_tokens + para_tokens <= max_tokens:
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens 

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def add_token_overlap(chunks: list[str], overlap_paragraphs: int = 1, model: str = "gpt-3.5-turbo", max_tokens: int = 6500):
    enc = tiktoken.encoding_for_model(model)
    result = []

    for i in range(len(chunks)):
        current = chunks[i]
        if i == 0:
            result.append(current)
            continue

        prev_paragraphs = chunks[i - 1].split('\n\n')[-overlap_paragraphs:]
        merged = '\n\n'.join(prev_paragraphs + [current])

        if len(enc.encode(merged)) <= max_tokens:
            result.append(merged)
        else:
            result.append(current)

    return result

def gen_flashcard(prompt: str, api_Key: str) -> str:
    # client = Together(api_key=apiKey)
    # res = client.chat.completions.create(
    #     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    #     messages=[{"role": "user", "content": prompt}],
    #     top_p=0.9,
    #     temperature=1.3,
    #
    # )

    headers = {
        "Authorization": f"Bearer {api_Key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": chuteModel,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 1.3
    }

    response = requests.post("https://llm.chutes.ai/v1/chat/completions", headers=headers, json=data)
    response = response.json()
    return response['choices'][0]['message']['content']


class PromptRequest(BaseModel):
    prompt: str

async def generate_card_async(chunks, instruction, max_requests_per_minute=6):
    """
    Process chunks with strict rate limiting.
    Sends up to max_requests_per_minute requests per minute.
    """
    results = []
    request_times = []

    total_chunks = len(chunks)
    print(f"Processing {total_chunks} chunks with rate limit of {max_requests_per_minute} requests per minute")

    i = 0
    while i < total_chunks:
        current_time = time.time()

        # Remove old timestamps beyond the 60-second window
        valid_times = []
        for t in request_times:
            if current_time - t < 60:
                valid_times.append(t)
        request_times = valid_times

        # Check if we've hit the rate limit
        if len(request_times) >= max_requests_per_minute:
            oldest_request_time = request_times[0]
            wait_duration = 60 - (current_time - oldest_request_time)

            if wait_duration > 0:
                print(f"Rate limit reached. Waiting {wait_duration:.1f} seconds... (chunk {i+1}/{total_chunks})")
                await asyncio.sleep(wait_duration)

                # Recheck the request_times after sleeping
                current_time = time.time()
                valid_times = []
                for t in request_times:
                    if current_time - t < 60:
                        valid_times.append(t)
                request_times = valid_times

        # Compose the prompt
        current_chunk = chunks[i]
        prompt = instruction + current_chunk

        print(f"Sending request for chunk {i+1}/{total_chunks} at {datetime.now().strftime('%H:%M:%S')}")

        try:
            request_times.append(time.time())

            result = await asyncio.to_thread(gen_flashcard, prompt, apiKey)

            if result == "**Rate limit exceeded.**":
                print(f"Rate limit hit for chunk {i+1}")
                request_times.pop()

                print("Waiting 60 seconds due to rate limit error...")
                await asyncio.sleep(60)

                request_times.append(time.time())
                result = await asyncio.to_thread(gen_flashcard, prompt)

            if "**Error:" in result:
                print(f"Error for chunk {i+1}: {result}")
            else:
                trimmed_result = result.strip()
                if 'Not enough relevant data to generate flashcards' not in trimmed_result:
                    results.append(trimmed_result)
                    print(f"✓ Successfully processed chunk {i+1}")
                else:
                    print(f"- Skipped chunk {i+1} (not enough data)")

        except Exception as e:
            print(f"Exception for chunk {i+1}: {e}")
            if request_times:
                request_times.pop()

        i += 1

    print(f"Completed processing. Generated {len(results)} flashcards from {total_chunks} chunks.")
    print (results)
    return results

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    with pdfplumber.open(file.file) as pdf:
        pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not any(c.isalnum() for c in pdf_text):
            return{"response": "No readable text found in the PDF."}

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    all_cards = []
    instruction = (
    "You are to act strictly as a flashcard generator.\n\n"
    "Given the content below, output **only** up to 10 Anki-style flashcards in the exact Markdown format shown.\n\n"
    "**Output Rules:**\n"
    "- Do NOT include any introductions, explanations, or summaries.\n"
    "- Do NOT write anything except the Q&A cards in the specified format.\n"
    "- Use only information **explicitly** stated in the content.\n"
    "- If there is not enough information, output exactly:\n"
    '**"Not enough relevant data to generate flashcards."**\n\n'
    "**Format (repeat up to 10 times):**\n"
    "Q: [Question]  \n"
    "A: [Answer]\n\n"
    "---\n\n"
    "CONTENT:\n"
)

    instruction_tokens = len(enc.encode(instruction))
    max_total_tokens = 6000
    max_output_tokens = 1024
    max_chunk_tokens = max_total_tokens - instruction_tokens - max_output_tokens

    raw_chunks = token_safe_chunking(pdf_text, max_tokens=max_chunk_tokens)
    chunks = add_token_overlap(raw_chunks, overlap_paragraphs=1)

    # for chunk in chunks:
    #     prompt = instruction + chunk
    #
    #     result = gen_flashcard(prompt)
    #     if '**\"Not enough relevant data to generate flashcards.\"**' not in result:
    #         all_cards.append(result)

    all_cards = await generate_card_async(chunks, instruction)

    if not all_cards:
        return {"response": "No relevant flashcard could be generated"}

    return {"response": "\n\n".join(all_cards)}

@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    response = gen_flashcard(data.prompt)
    return {"response": response}
