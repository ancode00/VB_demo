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
from bot_config import KB_FILE
import ffmpeg

import requests
import openai
from transformers import pipeline
from datetime import datetime
from statistics import mean

# For file parsing:
import io
import tempfile
from fastapi.middleware.cors import CORSMiddleware

# === Bot Readiness State and Config ===
app = FastAPI()
app.state.ready = False  # Bot is not ready until user confirms
app.state.session_config = {
    "bot_type": "incoming",   # incoming/outgoing
    "voice_id": "",
    "intro_line": "",
    "closing_line": ""
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# === KB/Prompt Management Start ===
SYSTEM_PROMPT_FILE = "system_prompt.txt"
KB_LOG_FILE = "kb_log.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("voicebot_api")

def log_kb_action(action, target, old_value, new_value):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "target": target,
        "old_value": old_value,
        "new_value": new_value
    }
    try:
        logs = []
        if os.path.exists(KB_LOG_FILE):
            with open(KB_LOG_FILE, "r") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(KB_LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write KB log: {e}")

def load_kb():
    try:
        with open(KB_FILE) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Could not load KB: {e}")
        return ""

def load_system_prompt():
    try:
        with open(SYSTEM_PROMPT_FILE) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Could not load system prompt: {e}")
        return ""

def save_kb(text):
    old = load_kb()
    with open(KB_FILE, "w") as f:
        f.write(text)
    log_kb_action("update", "kb", old, text)
    app.state.ready = False

def append_kb(text):
    old = load_kb()
    new = old + "\n" + text
    with open(KB_FILE, "w") as f:
        f.write(new)
    log_kb_action("append", "kb", old, new)
    app.state.ready = False

def reset_kb(text):
    old = load_kb()
    with open(KB_FILE, "w") as f:
        f.write(text)
    log_kb_action("reset", "kb", old, text)
    app.state.ready = False

def save_system_prompt(text):
    old = load_system_prompt()
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write(text)
    log_kb_action("update", "system.prompt", old, text)
    app.state.ready = False

def append_system_prompt(text):
    old = load_system_prompt()
    new = old + "\n" + text
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write(new)
    log_kb_action("append", "system.prompt", old, new)
    app.state.ready = False

def reset_system_prompt(text):
    old = load_system_prompt()
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write(text)
    log_kb_action("reset", "system.prompt", old, text)
    app.state.ready = False

@app.get("/get_kb")
def get_kb():
    return {"kb": load_kb()}

@app.get("/get_system_prompt")
def get_system_prompt():
    return {"system_prompt": load_system_prompt()}

@app.post("/update_kb")
async def update_kb(body: dict = Body(...)):
    save_kb(body.get("kb", ""))
    return {"status": "ok"}

@app.post("/append_kb")
async def append_kb_ep(body: dict = Body(...)):
    append_kb(body.get("kb", ""))
    return {"status": "ok"}

@app.post("/reset_kb")
async def reset_kb_ep(body: dict = Body(...)):
    reset_kb(body.get("kb", ""))
    return {"status": "ok"}

@app.post("/update_system_prompt")
async def update_system_prompt(body: dict = Body(...)):
    save_system_prompt(body.get("system_prompt", ""))
    return {"status": "ok"}

@app.post("/append_system_prompt")
async def append_system_prompt_ep(body: dict = Body(...)):
    append_system_prompt(body.get("system_prompt", ""))
    return {"status": "ok"}

@app.post("/reset_system_prompt")
async def reset_system_prompt_ep(body: dict = Body(...)):
    reset_system_prompt(body.get("system_prompt", ""))
    return {"status": "ok"}

@app.get("/kb_log")
def get_kb_log():
    try:
        with open(KB_LOG_FILE) as f:
            return {"log": json.load(f)}
    except:
        return {"log": []}

# ========== BOT SESSION CONFIGURATION ==========
@app.post("/set_config")
async def set_config(body: dict = Body(...)):
    """
    Set bot type, voice ID, intro line, and closing line for the session.
    """
    for key in ["bot_type", "voice_id", "intro_line", "closing_line"]:
        if key in body:
            app.state.session_config[key] = body[key]
    logger.info(f"Session config updated: {app.state.session_config}")
    return {"status": "ok", "session_config": app.state.session_config}

@app.post("/confirm_ready")
async def confirm_ready():
    app.state.ready = True
    logger.info("Bot readiness confirmed by user.")
    return {"status": "ready"}

# ========== KB FILE UPLOAD AND PARSE ENDPOINT ==========
from fastapi import File, UploadFile
import mimetypes

@app.post("/parse_kb_file")
async def parse_kb_file(file: UploadFile = File(...)):
    """
    Accepts a PDF, DOC, or DOCX file, parses text, and returns the text.
    """
    filename = file.filename
    content = await file.read()

    filetype = mimetypes.guess_type(filename)[0]
    text = ""

    try:
        if filename.lower().endswith(".pdf"):
            try:
                import PyPDF2
            except ImportError:
                return JSONResponse({"error": "PyPDF2 not installed on server"}, status_code=500)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif filename.lower().endswith((".doc", ".docx")):
            try:
                import docx
            except ImportError:
                return JSONResponse({"error": "python-docx not installed on server"}, status_code=500)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            doc = docx.Document(tmp_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            os.unlink(tmp_path)
        else:
            return JSONResponse({"error": "Unsupported file type"}, status_code=400)
        return {"text": text}
    except Exception as e:
        logger.error(f"File parse error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    conversation: str = Form("")
):
    # Block if not ready
    if not app.state.ready:
        return JSONResponse({"error": "Bot is not ready. Please confirm KB and prompt."}, status_code=400)

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

        # === Use latest KB and System Prompt ===
        kb_now = load_kb()
        system_prompt_now = load_system_prompt()

        # Pass session config to GPT/chat function if you want to use voice_id, intro_line, closing_line
        bot_reply = gpt_azure_chat(
            user_text,
            kb_now,
            system_prompt_now,
            history
        )

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

@app.get("/stream_tts")
async def stream_tts(text: str):
    # Use Voice ID from session if provided, else fallback to env
    voice_id = app.state.session_config.get("voice_id") or os.getenv("ELEVEN_VOICE_ID")
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    if not ELEVEN_API_KEY or not voice_id:
        return JSONResponse(
            {"error": "ElevenLabs configuration missing"},
            status_code=500
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ========== NEW CONVERSATION ANALYSIS ENDPOINT ==========
@app.post("/analyze")
async def analyze_conversation(body: dict = Body(...)):
    """
    Analyze the conversation using GPT and return a structured summary, lead interest, CSAT, and improvements.
    """
    try:
        conversation = body.get("conversation", [])
        if isinstance(conversation, str):
            try:
                conversation = json.loads(conversation)
            except Exception:
                pass

        # Compose prompt for GPT (make GPT always return ONLY a valid JSON)
        prompt = (
            "Analyze the following conversation for a summary (max 3 points), lead interest (1–10), csat (1–10), and improvements (1–2 lines). "
            "Always return: {\"summary\": [ ... ], \"lead_interest\": x, \"csat\": x, \"improvements\": \"...\"}. "
            "NO markdown, NO code block, NO explanation, ONLY pure JSON.\n"
            f"Conversation: {conversation}"
        )

        # You probably have a function for GPT, e.g., gpt_azure_chat:
        # (change as needed for your project)
        gpt_resp = gpt_azure_chat(prompt, "", "", [])

        # Try to parse
        analysis = None
        try:
            # If GPT returned a string containing JSON, parse it
            analysis = json.loads(gpt_resp)
        except Exception:
            # Fallback: extract JSON from string
            import re
            match = re.search(r'\{[\s\S]*\}', gpt_resp)
            if match:
                try:
                    analysis = json.loads(match.group(0))
                except Exception:
                    analysis = None

        if not analysis:
            # fallback if model fails
            analysis = {
                "summary": ["Unable to parse conversation summary due to model response format."],
                "lead_interest": None,
                "csat": None,
                "improvements": "Ensure the model returns structured JSON for analysis.",
                "raw": gpt_resp
            }
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        logger.error(f"Error during conversation analysis: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
