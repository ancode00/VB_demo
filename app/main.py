import os
import time
import json
import traceback
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from bot_helpers import elevenlabs_stt, gpt_azure_chat, elevenlabs_tts
from bot_config import KB_FILE, SYSTEM_PROMPT
import ffmpeg

import requests
import openai
from transformers import pipeline
from datetime import datetime
from statistics import mean

# ─── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("voicebot_api")

# ─── FastAPI Setup ────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ─── Load Knowledge Base ──────────────────────────────────────────
try:
    with open(KB_FILE) as f:
        KB = f.read()
    logger.info(f"Knowledge base loaded from {KB_FILE}")
except Exception as e:
    logger.error(f"Failed to load KB from {KB_FILE}: {e}")
    KB = "HPE ProLiant Servers\nCustomer Requests: ..."  # Fallback minimal KB

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANALYSIS_PROMPT = """Analyze this HPE ProLiant Servers sales conversation and provide:
1. A 5-point bullet summary
2. Lead interest level (Very High/High/Medium/Low/Very Low) and reason
3. CSAT prediction (Very Satisfied/Satisfied/Neutral/Dissatisfied/Very Dissatisfied) with confidence
4. Top improvement suggestion for the agent
5. Engagement score (0-100) based on message length, frequency, and intent clarity
6. Resolution status (Resolved/Unresolved/Escalated/Demo Scheduled) with explanation
7. Suggested follow-up action

Conversation:
{conversation}"""

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_with_openai(conversation_text: str) -> Dict[str, Any]:
    """Analyze conversation using OpenAI's API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        return analyze_with_fallback(conversation_text.split("\n"))

def analyze_with_fallback(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Enhanced fallback analysis with detailed insights"""
    try:
        # Get last 3 user messages for context
        recent_messages = [msg["content"] for msg in conversation[-3:] if isinstance(msg, dict) and msg.get("role") == "user"]
        combined_text = " ".join(recent_messages).lower() if recent_messages else ""
        
        # Enhanced sentiment analysis
        sentiment = {"label": "NEUTRAL", "score": 0.5}
        if recent_messages:
            try:
                sentiment_result = sentiment_analyzer(recent_messages[-1])
                sentiment = sentiment_result[0] if isinstance(sentiment_result, list) else sentiment_result
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Enhanced topic detection
        topics = set()
        topic_keywords = {
            "HPE ProLiant": ["proliant", "server"],
            "Server Models": ["dl", "gen", "ml", "bl"],
            "Pricing": ["price", "cost", "budget"],
            "Configuration": ["config", "spec", "memory", "cpu", "storage"],
            "Support": ["support", "warranty", "help"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                topics.add(topic)
                
        # Calculate interest score
        interest_factors = {
            "message_length": sum(len(msg) for msg in recent_messages),
            "question_count": sum(1 for msg in recent_messages if "?" in msg),
            "specific_terms": sum(1 for term in ["proliant", "dl", "gen", "spec"] if term in combined_text)
        }
        
        interest_score = min(
            (interest_factors["message_length"] / 100) + 
            (interest_factors["question_count"] * 10) +
            (interest_factors["specific_terms"] * 15), 
            100
        )
        
        interest_level = (
            "Very High" if interest_score > 80 else
            "High" if interest_score > 60 else
            "Medium" if interest_score > 40 else
            "Low"
        )
        
        # Generate meaningful summary points
        summary_points = [
            f"Customer discussed {len(topics)} main topics" if topics else "General inquiry",
            f"Conversation shows {interest_level.lower()} interest level",
            f"Recent sentiment: {sentiment['label']} (confidence: {sentiment['score']:.0%})",
            "Key topics: " + ", ".join(topics) if topics else "No specific topics identified",
            f"{len(recent_messages)} user messages analyzed"
        ]
        
        return {
            "summary": summary_points,
            "interest_level": interest_level,
            "csat_prediction": (
                "Very Satisfied" if sentiment["score"] > 0.9 else
                "Satisfied" if sentiment["score"] > 0.7 else
                "Neutral" if sentiment["score"] > 0.4 else
                "Dissatisfied"
            ),
            "improvement_suggestion": (
                "Ask about specific server needs" if "HPE ProLiant" in topics else
                "Discuss pricing options" if "Pricing" in topics else
                "Provide technical specifications" if "Configuration" in topics else
                "General sales approach recommended"
            ),
            "engagement_score": int(interest_score),
            "resolution_status": (
                "Demo Scheduled" if any(x in combined_text for x in ["demo", "meeting"]) else
                "Follow-up Needed" if interest_score > 50 else
                "Unresolved"
            ),
            "follow_up_action": (
                "Schedule technical demo" if interest_level in ["Very High", "High"] else
                "Send product information" if interest_level == "Medium" else
                "General follow-up email"
            ),
            "sentiment": sentiment["label"],
            "confidence": float(sentiment["score"]),
            "main_topics": list(topics) if topics else ["General Inquiry"]
        }
        
    except Exception as e:
        logger.error(f"Enhanced fallback analysis failed: {traceback.format_exc()}")
        return {
            "error": str(e),
            "summary": ["Analysis temporarily unavailable"],
            "interest_level": "Unknown",
            "csat_prediction": "Unknown"
        }

class Message(BaseModel):
    role: str
    content: str

class ConversationIn(BaseModel):
    conversation: List[Message]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    conversation: str = Form("")
):
    logger.info("Received /ask POST request")
    t0 = time.time()
    webm_path = "app/static/user_input.webm"
    wav_path = "app/static/user_input.wav"

    try:
        # Save uploaded audio
        with open(webm_path, "wb") as f:
            file_bytes = await audio.read()
            f.write(file_bytes)
        t1 = time.time()

        # Convert to WAV format
        ffmpeg.input(webm_path).output(
            wav_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar='16000'
        ).overwrite_output().run(quiet=True)
        t2 = time.time()

        # Speech-to-text
        stt_result_path = elevenlabs_stt(wav_path)
        with open(stt_result_path) as f:
            stt_json = json.load(f)
        user_text = stt_json.get("text") or stt_json.get("transcript", "")
        t3 = time.time()

        # Process conversation history
        history = []
        if conversation:
            try:
                history = json.loads(conversation)
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding conversation history: {e}")

        # Get bot response
        bot_reply = gpt_azure_chat(user_text, KB, SYSTEM_PROMPT, history)
        t4 = time.time()

        # Update conversation history
        updated_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_reply}
        ]

        logger.info(
            f"Request processed in {t4-t0:.2f}s (audio: {t1-t0:.2f}s, "
            f"convert: {t2-t1:.2f}s, stt: {t3-t2:.2f}s, gpt: {t4-t3:.2f}s)"
        )

        return JSONResponse({
            "transcript": user_text,
            "bot_reply": bot_reply,
            "conversation": json.dumps(updated_history)
        })

    except Exception as e:
        logger.error(f"Exception in /ask endpoint: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze")
async def analyze_conversation(payload: ConversationIn):
    try:
        conversation = [msg.dict() for msg in payload.conversation]
        if not conversation:
            return {"status": "error", "message": "No conversation data received."}

        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation[-6:]
        )

        analysis_result = (
            analyze_with_openai(conversation_text)
            if OPENAI_API_KEY
            else analyze_with_fallback(conversation)
        )

        return {
            "status": "success",
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(conversation),
            "main_topic": (
                analysis_result.get("main_topics", ["General Inquiry"])[0]
                if analysis_result.get("main_topics")
                else "General Inquiry"
            )
        }
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@app.get("/tts_intro")
async def tts_intro():
    intro_text = os.getenv("BOT_INTRO_TEXT", "Hello, I'm calling from HPE about your server needs. Do you have a moment to talk?")
    intro_audio_path = "app/static/intro.wav"
    try:
        elevenlabs_tts(intro_text, intro_audio_path)
        return JSONResponse({
            "audio_url": "/static/intro.wav",
            "intro_text": intro_text
        })
    except Exception as e:
        logger.error(f"Intro TTS failed: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/stream_tts")
async def stream_tts(text: str):
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        return JSONResponse(
            {"error": "ElevenLabs configuration missing"},
            status_code=500
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.7
        }
    }

    def generate_audio():
        try:
            with requests.post(url, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
        except Exception as e:
            logger.error(f"TTS streaming error: {traceback.format_exc()}")
            raise

    return StreamingResponse(
        generate_audio(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"}
    )
