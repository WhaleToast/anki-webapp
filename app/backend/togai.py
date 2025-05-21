import os
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

def gen_flashcard(prompt: str) -> str:
    client = Together(api_key=apiKey)
    print("PROMPT HERE!!!!: ", prompt)
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
    prompt = (
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
        f"{pdf_text[:23000]}"
)
    response = gen_flashcard(prompt)
    return {"response": response}


@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    response = gen_flashcard(data.prompt)
    return {"response": response}
