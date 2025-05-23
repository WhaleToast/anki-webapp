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
    print(response)
    return response['choices'][0]['message']['content']


class PromptRequest(BaseModel):
    prompt: str

async def generate_card_async(chunks, instruction, max_requests_per_minute=35, batch_size=10):
    """
    Process chunks in batches with strict rate limiting.
    Sends up to batch_size requests concurrently, respecting max_requests_per_minute.
    """
    results = []
    request_times = []
    total_chunks = len(chunks)
    
    print(f"Processing {total_chunks} chunks in batches of {batch_size} with rate limit of {max_requests_per_minute} requests per minute")
    
    # Process chunks in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]
        
        current_time = time.time()
        
        # Remove old timestamps beyond the 60-second window
        request_times = [t for t in request_times if current_time - t < 60]
        
        # Check if we need to wait for rate limit
        requests_needed = len(batch_chunks)
        available_slots = max_requests_per_minute - len(request_times)
        
        if available_slots < requests_needed:
            if request_times:
                oldest_request_time = request_times[0]
                wait_duration = 60 - (current_time - oldest_request_time) + 1  # +1 for safety
                if wait_duration > 0:
                    print(f"Rate limit reached. Waiting {wait_duration:.1f} seconds for batch {batch_start//batch_size + 1}")
                    await asyncio.sleep(wait_duration)
                    # Refresh request times after waiting
                    current_time = time.time()
                    request_times = [t for t in request_times if current_time - t < 60]
        
        # Create tasks for the current batch
        tasks = []
        for i, chunk in enumerate(batch_chunks):
            chunk_index = batch_start + i
            prompt = instruction + chunk
            task = asyncio.to_thread(gen_flashcard, prompt, apiKey)
            tasks.append(task)
        
        # Record request times for rate limiting
        batch_request_time = time.time()
        request_times.extend([batch_request_time] * len(batch_chunks))
        
        print(f"Sending batch {batch_start//batch_size + 1} with {len(batch_chunks)} requests at {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute batch concurrently
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                chunk_index = batch_start + i
                if isinstance(result, Exception):
                    print(f"Exception for chunk {chunk_index + 1}: {result}")
                elif result == "**Rate limit exceeded.**":
                    print(f"Rate limit hit for chunk {chunk_index + 1}")
                    # Could implement retry logic here
                elif "**Error:" in str(result):
                    print(f"Error for chunk {chunk_index + 1}: {result}")
                else:
                    trimmed_result = str(result).strip()
                    if 'Not enough relevant data to generate flashcards' not in trimmed_result:
                        results.append(trimmed_result)
                        print(f"✓ Successfully processed chunk {chunk_index + 1}")
                    else:
                        print(f"- Skipped chunk {chunk_index + 1} (not enough data)")
                        
        except Exception as e:
            print(f"Batch processing error: {e}")
    
    print(f"Completed processing. Generated {len(results)} flashcards from {total_chunks} chunks.")
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
