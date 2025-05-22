import os
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

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    with pdfplumber.open(file.file) as pdf:
        pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if not pdf_text.strip():
            return{"response": "No readable text found in the PDF."}

    # raw_chunks = token_safe_chunking(pdf_text, max_tokens=6500)
    # print(raw_chunks)
    # chunks = add_token_overlap(raw_chunks, overlap_paragraphs=1)


    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    all_cards = []
    instruction = (
    "Create 10 unique and high-quality Anki flashcards based on the following content ONLY.\n\n"
    "Write ONLY the flashcard and markdown. Do NOT write any other text.\n\n"
    "Do NOT use any external or prior knowledge. If the provided content does not contain enough information to create five unique flashcards, "
    "reply with exactly this sentence:\n\n"
    "**\"Not enough relevant data to generate flashcards.\"**\n\n"
    "Each card must focus on a distinct, concrete fact or concept explicitly mentioned in the input. Do not invent or assume anything. Add an explanation where applicable, from provided data ONLY.\n\n"
    "Use the following format for each card, using markdown for readability:\n\n"
    "Q: [question]  \n"
    "A: [answer]\n\n"
    "Here is the content:\n"
)

    instruction_tokens = len(enc.encode(instruction))
    max_total_tokens = 6000
    max_output_tokens = 1024
    max_chunk_tokens = max_total_tokens - instruction_tokens - max_output_tokens

    raw_chunks = token_safe_chunking(pdf_text, max_tokens=max_chunk_tokens)
    chunks = add_token_overlap(raw_chunks, overlap_paragraphs=1)

    for chunk in chunks:
        prompt = instruction + chunk

        result = gen_flashcard(prompt)
        if '**\"Not enough relevant data to generate flashcards.\"**' not in result:
            all_cards.append(result)

    if not all_cards:
        return {"response": "No relevant flashcard could be generated"}

    return {"response": "\n\n".join(all_cards)}

@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    response = gen_flashcard(data.prompt)
    return {"response": response}
