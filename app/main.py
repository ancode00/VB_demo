import os
import time
import json
import traceback
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
load_dotenv()

from bot_helpers import elevenlabs_stt, gpt_azure_chat, elevenlabs_tts
from bot_config import KB_FILE, SYSTEM_PROMPT
import ffmpeg

import openai
from transformers import pipeline  # Hugging Face fallback
from datetime import datetime

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
    KB = ""

# ─── Analysis AI/Fallback Setup ──────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANALYSIS_PROMPT = """Analyze this conversation and provide:
1. A 5-point bullet summary
2. Lead interest level (High/Medium/Low) and reason
3. CSAT prediction (Satisfied/Neutral/Dissatisfied) with confidence
4. Top improvement suggestion for the agent

Conversation:
{conversation}"""

sentiment_analyzer = pipeline("sentiment-analysis")

def extract_main_topic(text):
    topics = ['pricing', 'support', 'features', 'availability', 'demo']
    return next((t for t in topics if t in text.lower()), 'general inquiry')

def parse_ai_response(text):
    return {"raw": text}

def analyze_with_openai(conversation_text):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You're an expert conversation analyst. Provide crisp, structured analysis."
        }, {
            "role": "user",
            "content": ANALYSIS_PROMPT.format(conversation=conversation_text)
        }],
        temperature=0.3
    )
    return parse_ai_response(response.choices[0].message['content'])

def analyze_with_fallback(conversation):
    if not conversation or not isinstance(conversation, list):
        logger.error("Fallback analysis received empty or invalid conversation.")
        return {"error": "No valid conversation data received."}
    last_user_msg = next((msg['content'] for msg in reversed(conversation) if msg.get('role') == 'user'), '')
    if not last_user_msg:
        logger.error("Fallback analysis: no user messages found.")
        return {"error": "No user messages in conversation."}
    sentiment = sentiment_analyzer(last_user_msg)[0]
    csat = "Satisfied" if sentiment['label'] == 'POSITIVE' else "Neutral" if sentiment['score'] < 0.7 else "Dissatisfied"
    interest_keywords = {
        'price': 'High', 'cost': 'High', 'buy': 'High',
        'info': 'Medium', 'detail': 'Medium', 'how': 'Medium',
        'hello': 'Low', 'hi': 'Low', 'thanks': 'Low'
    }
    interest = next((v for k, v in interest_keywords.items() if k in last_user_msg.lower()), 'Low')
    return {
        "summary": [
            "User initiated conversation",
            f"Key topic: {extract_main_topic(last_user_msg)}",
            f"Sentiment detected: {sentiment['label']} ({sentiment['score']:.0%})",
            "Agent responded to query",
            "Conversation concluded"
        ],
        "lead_interest": f"{interest} Intent (Keyword: {extract_main_topic(last_user_msg)})",
        "csat": f"{csat} ({sentiment['score']:.0%} confidence)",
        "improvements": "Agent could ask more qualifying questions" if interest == 'High' else "Agent could provide more detailed information"
    }

# Pydantic models for robust parsing & logging
class Message(BaseModel):
    role: str
    content: str

class ConversationIn(BaseModel):
    conversation: List[Message]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("Rendering index.html for /")
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
    bot_audio_path = "app/static/bot_reply.wav"

    try:
        logger.info("Saving uploaded audio file...")
        with open(webm_path, "wb") as f:
            file_bytes = await audio.read()
            f.write(file_bytes)
        t1 = time.time()
        logger.info(f"Audio saved as {webm_path} ({t1 - t0:.2f}s, {len(file_bytes)} bytes)")

        logger.info("Converting to 16kHz, mono, PCM16 WAV via ffmpeg...")
        ffmpeg.input(webm_path).output(
            wav_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar='16000'
        ).overwrite_output().run(quiet=True)
        t2 = time.time()
        logger.info(f"ffmpeg conversion complete ({t2 - t1:.2f}s)")

        os.system(f"file {wav_path}")

        logger.info("Transcribing audio with ElevenLabs STT...")
        stt_result_path = elevenlabs_stt(wav_path)
        with open(stt_result_path) as f:
            stt_json = json.load(f)
        user_text = stt_json.get("text") or stt_json.get("transcript")
        logger.info(f"User said: {user_text}")
        t3 = time.time()
        logger.info(f"Transcription received: '{user_text}' ({t3 - t2:.2f}s)")

        history = []
        if conversation:
            try:
                history = json.loads(conversation)
                logger.info(f"Decoded conversation history (len={len(history)})")
            except Exception as e:
                logger.warning(f"Error decoding conversation history: {e} (raw={conversation[:200]})")
        else:
            logger.info("No conversation history provided.")

        logger.debug(f"Original conversation history: {history}")

        if len(history) > 4:
            logger.info(f"Trimming conversation history from {len(history)} to last 4 turns")
            history = history[-4:]
        logger.debug(f"Trimmed conversation history: {history}")

        logger.info("Sending transcript to Azure GPT...")
        bot_reply = gpt_azure_chat(user_text, KB, SYSTEM_PROMPT, history)
        t4 = time.time()
        logger.info(f"Bot reply received: '{bot_reply}' ({t4 - t3:.2f}s)")

        logger.info("Sending bot reply to ElevenLabs TTS...")
        elevenlabs_tts(bot_reply, bot_audio_path)
        t5 = time.time()
        logger.info(f"TTS audio generated at {bot_audio_path} ({t5 - t4:.2f}s)")

        history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_reply}
        ]
        logger.info(f"Updated conversation history: {len(history)} turns")

        logger.info(
            "[TIMING] upload=%.2fs | convert=%.2fs | stt=%.2fs | gpt=%.2fs | tts=%.2fs | total=%.2fs",
            t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0
        )

        return JSONResponse({
            "transcript": user_text,
            "bot_reply": bot_reply,
            "audio_url": "/static/bot_reply.wav",
            "conversation": json.dumps(history)
        })

    except Exception as e:
        logger.error("----- Exception in /ask endpoint -----")
        logger.error(traceback.format_exc())
        logger.error("----- End Exception -----")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze")
async def analyze_conversation(payload: ConversationIn):
    try:
        logger.info(f"Received /analyze request: {payload}")
        conversation = [msg.dict() for msg in payload.conversation]
        if not conversation:
            logger.error("No conversation provided to /analyze.")
            return {"status": "error", "message": "No conversation data received."}

        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation[-6:]
        )
        logger.info(f"Analysis input text: {conversation_text}")

        if OPENAI_API_KEY:
            analysis = analyze_with_openai(conversation_text)
        else:
            analysis = analyze_with_fallback(conversation)

        logger.info(f"Analysis output: {analysis}")
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Error during conversation analysis:")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/tts_intro")
async def tts_intro():
    """
    Returns TTS audio_url and the intro text from env.
    """
    intro_text = os.getenv("BOT_INTRO_TEXT", "Hello! I am your assistant.")
    intro_audio_path = "app/static/intro.wav"
    try:
        logger.info(f"Generating intro TTS: '{intro_text}'")
        elevenlabs_tts(intro_text, intro_audio_path)
        logger.info(f"Intro TTS generated at {intro_audio_path}")
        return JSONResponse({"audio_url": "/static/intro.wav", "intro_text": intro_text})
    except Exception as e:
        logger.error("Intro TTS generation failed:")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
