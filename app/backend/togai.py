from logging import exception
import os
import genanki
import random
import asyncio
import time
import re
import tiktoken
import requests
from pathlib import Path
from together import Together
from dotenv import load_dotenv 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pdfplumber 
from fastapi import File, UploadFile
from datetime import datetime

envPath = Path("../../secrets/.env")
load_dotenv(dotenv_path=envPath)
apiKey = os.getenv("chutes")
chuteModel = "deepseek-ai/DeepSeek-V3-0324"
global_max_tokens = 19000


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)
def split_paragraphs(text:str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}|\n(?=\S)', text) if p.strip()]

def token_safe_chunking(text: str, max_tokens: int = global_max_tokens, model: str = "gpt-3.5-turbo") -> list[str]:
    enc = tiktoken.encoding_for_model(model)
    paragraphs = split_paragraphs(text)    
    print(max_tokens)
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

def add_token_overlap(chunks: list[str], overlap_paragraphs: int = 1, model: str = "gpt-3.5-turbo", max_tokens: int = global_max_tokens):
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
    # print(response)
    return response['choices'][0]['message']['content']


class PromptRequest(BaseModel):
    prompt: str

async def generate_card_async(chunks, instruction, max_requests_per_minute=20, batch_size=10):
    """
    Process chunks in batches with strict rate limiting.
    Sends up to batch_size requests concurrently, respecting max_requests_per_minute.
    
    Parameters:
    - chunks: List of text pieces to convert into flashcards
    - instruction: The prompt/instruction to tell the AI how to make flashcards
    - max_requests_per_minute: How many API calls we can make per minute (default 35)
    - batch_size: How many requests to send at the same time (default 10)
    """
    
    # Initialize our storage containers
    results = []           # Will store all the successful flashcards we generate
    request_times = []     # Keeps track of when we made each API request (for rate limiting)
    total_chunks = len(chunks)  # Count how many text chunks we need to process
    
    # Tell the user what we're about to do
    print(f"Processing {total_chunks} chunks in batches of {batch_size} with rate limit of {max_requests_per_minute} requests per minute")
    
    # MAIN LOOP: Process chunks in batches
    # This loop goes through chunks in groups (batches) instead of one by one
    # range(start, stop, step) - starts at 0, goes to total_chunks, jumps by batch_size
    for batch_start in range(0, total_chunks, batch_size):
        
        # Figure out which chunks belong to this batch
        batch_end = min(batch_start + batch_size, total_chunks)  # Don't go past the end
        batch_chunks = chunks[batch_start:batch_end]  # Get the actual chunk data for this batch
        
        # RATE LIMITING SECTION
        # We need to make sure we don't send too many requests too quickly
        current_time = time.time()  # Get current timestamp
        
        # Clean up old request timestamps (only keep requests from last 60 seconds)
        # This is like a "sliding window" - we only care about recent requests
        request_times = [t for t in request_times if current_time - t < 60]
        
        # Check if we have room for more requests within our rate limit
        requests_needed = len(batch_chunks)  # How many requests this batch will make
        available_slots = max_requests_per_minute - len(request_times)  # How many slots we have left
        
        # If we don't have enough available slots, we need to wait
        if available_slots < requests_needed:
            if request_times:  # If we have previous requests to base timing on
                oldest_request_time = request_times[0]  # Find the oldest request in our window
                # Calculate how long to wait until that old request "expires" from our 60-second window
                wait_duration = 60 - (current_time - oldest_request_time) + 1  # +1 for safety buffer
                
                if wait_duration > 0:  # If we actually need to wait
                    print(f"Rate limit reached. Waiting {wait_duration:.1f} seconds for batch {batch_start//batch_size + 1}")
                    await asyncio.sleep(wait_duration)  # Actually pause execution
                    
                    # After waiting, update our timing info
                    current_time = time.time()
                    request_times = [t for t in request_times if current_time - t < 60]
        
        # PREPARE THE BATCH OF REQUESTS
        # Create a list of "tasks" (async operations) for this batch
        tasks = []
        for i, chunk in enumerate(batch_chunks):
            chunk_index = batch_start + i  # Calculate the overall index of this chunk
            prompt = instruction + chunk   # Combine the instruction with the text chunk
            # Create an async task that will call gen_flashcard function
            task = asyncio.to_thread(gen_flashcard, prompt, apiKey)
            tasks.append(task)
        
        # Record when we're making these requests (for future rate limiting)
        batch_request_time = time.time()
        # Add one timestamp for each request in this batch
        request_times.extend([batch_request_time] * len(batch_chunks))
        
        # Tell user we're sending this batch
        print(f"Sending batch {batch_start//batch_size + 1} with {len(batch_chunks)} requests at {datetime.now().strftime('%H:%M:%S')}")
        
        # EXECUTE THE BATCH
        # Run all tasks in this batch simultaneously and wait for all to complete
        try:
            # asyncio.gather runs multiple async operations concurrently
            # return_exceptions=True means if one fails, others keep running
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # PROCESS THE RESULTS
            # Go through each result and handle different types of responses
            for i, result in enumerate(batch_results):
                chunk_index = batch_start + i  # Calculate which original chunk this result belongs to
                
                # Handle different types of results:
                
                if isinstance(result, Exception):
                    # Something went wrong with this specific request
                    print(f"Exception for chunk {chunk_index + 1}: {result}")
                    
                elif "Too many requests" in str(result):
                    # The API told us we hit the rate limit
                    print(f"Rate limit hit for chunk {chunk_index + 1}")
                    # Could implement retry logic here if needed
                    
                elif "**Error:" in str(result):
                    # The API returned some other kind of error
                    print(f"Error for chunk {chunk_index + 1}: {result}")
                    
                else:
                    print(result)
                    # This looks like a successful result!
                    trimmed_result = str(result).strip()  # Remove extra whitespace
                    
                    # Check if the result is actually useful
                    if 'Not enough relevant data to generate flashcards' not in trimmed_result:
                        # Good result - add it to our collection
                        results.append(trimmed_result)
                        print(f"✓ Successfully processed chunk {chunk_index + 1}")
                    else:
                        # The AI said this chunk didn't have enough good content
                        print(f"- Skipped chunk {chunk_index + 1} (not enough data)")
                        
        except Exception as e:
            # If something went wrong with the entire batch
            print(f"Batch processing error: {e}")
    
    # ALL DONE!
    print(f"Completed processing. Generated {len(results)} flashcards from {total_chunks} chunks.")
    return results  # Return all the successfully generated flashcards    
 
def parse_flashcards_to_anki(llm_output, my_model, my_deck, deck_id):
    """
    Parse LLM output containing Q: and A: pairs and convert to Anki notes.
    
    Args:
        llm_output (str): The raw text output from the LLM
        my_model: The genanki model to use for creating notes
        my_deck: The genanki deck to use for generating the deck
    
    Returns:
        Nothing
    """
    # This pattern looks for Q: followed by text, then A: followed by text
    pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
    matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        # Clean up the text (remove extra whitespace and newlines)
        question = question.strip().replace('\n', ' ')
        answer = answer.strip().replace('\n', ' ')
        if not question or not answer:
            continue
        if question.lower() == answer.lower():
            print(f"Skipped duplicate Q&A: {question}")
            continue

        # non_ascii_ratio = sum(1 for c in answer if ord(c) > 127) / len(answer)
        # if non_ascii_ratio > 0.3: #More than 30%
        #     print(f"Skipped garbled answer for: {question} {answer}")

        error_patterns = [
            "Please wait",
            "Sorry令",
            "Recently deleted",
            "研究表明",  # Chinese characters when not expected
            "μηδὲν",     # Greek characters when not expected
        ]

        if any(pattern in answer for pattern in error_patterns):
            print(f"Skipped error-containing answer for: {question} {answer}")
            continue


        # Skip empty questions or answers
        note = genanki.Note(
            model=my_model,
            fields=[question, answer]
        )
        my_deck.add_note(note)

    os.makedirs("decks", exist_ok=True)
    filename = f"decks/{deck_id}.apkg"
    genanki.Package(my_deck).write_to_file(filename)

    return filename

@app.get("/api/download-deck/{deck_id}")
async def download_deck(deck_id: str):
    try:
        deck_path = Path("../backend/decks") / f"{deck_id}.apkg"

        if not deck_path.exists():
            raise HTTPException(status_code=404, detail="Deck not found")

        return FileResponse(
            path=deck_path,
            filename=f"{deck_id}.apkg",
            media_type="application/octet-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Download failed")
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    random_model_id = random.randrange(1 << 30, 1 << 31)
    random_deck_id = random.randrange(1 << 30, 1 << 31)
    anki_deck = genanki.Deck(
        random_deck_id,
        f'Deck_{random_deck_id}') 
    anki_model = genanki.Model(
        random_model_id, 
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ]
    )

    with pdfplumber.open(file.file) as pdf:
        pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not any(c.isalnum() for c in pdf_text):
            return{"response": "No readable text found in the PDF."}

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    all_cards = []
    instruction = (
        "You are an expert flashcard generator that adapts to any type of content.\n\n"
        "TASK: Generate up to 20 high-quality flashcards from the content below.\n\n"
        "CORE PRINCIPLES:\n"
        "- Create questions that promote active recall and retention\n"
        "- Focus on information that learners need to retrieve quickly\n"
        "- One clear concept per flashcard\n"
        "- Concise but complete answers\n"
        "- Adapt question style to match the content type\n\n"
        "AUTO-ADAPT TO CONTENT TYPE:\n"
        "- If Academic/Technical: Focus on definitions, processes, relationships, applications\n"
        "- If Language Content: Focus on vocabulary, grammar, usage, translations\n"
        "- If Historical/Factual: Focus on events, people, dates, causes, effects\n"
        "- If Procedural: Focus on steps, methods, when/how to apply\n"
        "- If Conceptual: Focus on understanding, comparisons, principles\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY the flashcards in the exact format shown below\n"
        "- No introductions, explanations, summaries, or extra text\n"
        "- Use only information explicitly stated in the content\n"
        "- Skip trivial or overly obvious information\n"
        "- Prioritize frequently needed information over obscure details\n"
        "- Do NOT use markdown for formatting in any way, shape or form. Use ONLY plain text\n"
        "- If insufficient quality content, output exactly: 'Not enough relevant data to generate flashcards.'\n\n"
        "REQUIRED FORMAT:\n"
        "Q: [Clear, specific question]\n"
        "A: [Concise, complete answer]\n\n"
        "Q: [Clear, specific question]\n"
        "A: [Concise, complete answer]\n\n"
        "[Continue for up to 20 cards]\n\n"
        "---\n\n"
        "CONTENT:\n"
    )

    instruction_tokens = len(enc.encode(instruction))
    max_total_tokens = global_max_tokens
    max_output_tokens = 2042
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

    # return {"response": "\n\n".join(all_cards)}
    response = "\n\n".join(all_cards)
    
    parse_flashcards_to_anki(response, anki_model, anki_deck, random_deck_id)
    

    return {
        "response": response,
        "deck_id": str(random_deck_id)
    }

@app.post("/generate")
def generate_anki_cards(data: PromptRequest):
    response = gen_flashcard(data.prompt)
    return {"response": response}
