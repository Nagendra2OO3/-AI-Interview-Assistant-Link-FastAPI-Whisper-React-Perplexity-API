from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import whisper
import tempfile
from textblob import TextBlob
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai.api_key)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup
model = whisper.load_model("base")

@app.get("/")
def read_root():
    return {"message": "It works!"}

@app.post("/interview/audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe audio using Whisper
        result = model.transcribe(tmp_path)
        transcript = result["text"]

        # Analyze sentiment with TextBlob
        blob = TextBlob(transcript)
        sentiment = blob.sentiment

        # Generate a follow-up question with OpenAI
        prompt = f"Candidate said: '{transcript}'. Listen to this answer: What is a thoughtful follow-up question?"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        follow_up = response.choices[0].message.content

        # Return response to frontend
        return {
            "transcript": transcript,
            "sentiment": {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            },
            "follow_up": follow_up
        }

    except Exception as e:
        return {"error": str(e)}
