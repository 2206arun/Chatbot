import os
import httpx
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv

# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="LLaMA Chatbot API", version="1.1")


@app.get("/", response_class=PlainTextResponse)
def welcome():
    return (
        "Welcome to the LLaMA Chatbot API!\n"
        "To chat, use: /chat?message=Your+message+here"
    )


@app.get("/chat")
async def chat(message: str = Query(..., min_length=1, description="Your message to the chatbot")):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",  # Best for chatbot-style interaction
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

    data = response.json()
    reply = data["choices"][0]["message"]["content"].strip()

    return JSONResponse(content={"user": message, "bot": reply})
