import os
import asyncio
import time
import re
import tiktoken
from pathlib import Path
from together import Together
from dotenv import load_dotenv 
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber 
from fastapi import File, UploadFile

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

def gen_flashcard(prompt: str) -> str:
    client = Together(api_key=apiKey)
    res = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        top_p=0.9,
        temperature=1.3,

    )

    return res.choices[0].message.content



class PromptRequest(BaseModel):
    prompt: str

async def generate_card_async(chunks, instruction, batch_size=6, delay=60):
    results = []
    i = 0

    while i < len(chunks):
        batch_start_time = time.time()
        print(f"Starting batch at chunk {i}, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        tasks = []
        for _ in range(batch_size):
            if i >= len(chunks):
                break
            prompt = instruction + chunks[i]
            task = asyncio.create_task(asyncio.to_thread(gen_flashcard, prompt))
            tasks.append(task)

            i += 1
            print(i)

        # Start the delay countdown immediately
        delay_task = asyncio.create_task(asyncio.sleep(delay))

        # Wait for the delay to finish before starting the next batch
        await delay_task

        # Collect results from the current batch
        batch_results = await asyncio.gather(*tasks)

        # Process results
        for j, result in enumerate(batch_results):
            if result == "**Rate limit exceeded.**":
                print(f"Rate limit hit at chunk {i-len(batch_results)+j}, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                continue
            if "**Error:" in result:
                print(f"Error at chunk {i-len(batch_results)+j}: {result}")
                continue
            if '**"Not enough relevant data to generate flashcard."**' not in result.strip():
                results.append(result.strip())

        # Ensure the next batch starts exactly 60 seconds after the previous batch started
        elapsed = time.time() - batch_start_time
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

    print(f"Total chunks processed: {i}")
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
        "Based solely on the content provided below, generate up to 10 high-quality Anki-style flashcards.\n\n"
        "Each flashcard must:\n"
        "- Be based only on facts, definitions, or concepts **explicitly stated** in the text.\n"
        "- Contain **no assumptions** or external knowledge.\n"
        "- Be phrased clearly and concisely for effective spaced repetition.\n"
        "- Include brief **explanations or clarifications** if they are present in the source text.\n"
        "- Be unique — avoid rewording the same idea multiple times.\n\n"
        "If the content is too limited to generate meaningful flashcards, respond with exactly:\n"
        "**\"Not enough relevant data to generate flashcards.\"**\n\n"
        "Use **Markdown format** like this:\n"
        "Q: [Question]  \n"
        "A: [Answer]\n\n"
        "--- Do not include any headings, intro text, or explanations outside of the Q&A format. ---\n\n"
        "Here is the content:\n"
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
