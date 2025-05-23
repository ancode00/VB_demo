import os
import time
import requests
import logging
import json

from bot_config import *
from dotenv import load_dotenv

# ─── Configure Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ensure .env variables are loaded, even if this is called directly
load_dotenv()
logging.info("Loaded environment variables from .env")

# === Helper to load env vars at call time (for robustness) ===
def get_env_var(key, default=None):
    value = os.getenv(key, default)
    if value is None:
        logging.warning(f"Environment variable {key} is not set!")
    return value

# ─── ElevenLabs Speech-to-Text ────────────────────────────────────
def elevenlabs_stt(audio_path):
    ELEVEN_API_KEY = get_env_var("ELEVEN_API_KEY")
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVEN_API_KEY}
    out_path = "user_transcript.json"
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model_id": "scribe_v1"}
        logging.info(f"Sending {audio_path} to ElevenLabs STT (model_id='scribe_v1')...")
        r = requests.post(url, headers=headers, files=files, data=data)
        logging.debug(f"ElevenLabs STT raw response: {r.status_code} {r.text}")
        try:
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Failed to get STT from ElevenLabs: {e}")
            return None
        with open(out_path, "wb") as out_f:
            out_f.write(r.content)
    logging.info(f"STT output saved to: {out_path}")
    return out_path

# ─── Azure Chat ───────────────────────────────────────────────────
def gpt_azure_chat(prompt, kb, system_prompt, messages):
    # Only send last 2 user-bot turns (4 messages)
    short_history = messages[-4:] if len(messages) > 4 else messages

    concise_prompt = (
        f"{system_prompt}\n\n"
        "Knowledge Base:\n" + kb + "\n\n"
        "Be concise, respond in 2–3 sentences max."
    )
    msgs = [{"role": "system", "content": concise_prompt}] + short_history + [{"role": "user", "content": prompt}]
    headers = {
        "api-key": get_env_var("AZURE_OPENAI_API_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "messages": msgs,
        "temperature": 0.6,
        "max_tokens": 100   # You can drop to 80 for even snappier replies
    }
    endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    logging.info(f"Sending {len(msgs)} messages to Azure GPT (max_tokens={data['max_tokens']})")

    # ========= DEBUG PAYLOAD =========
    print("====[DEBUG] GPT CALL PAYLOAD====")
    print("System prompt:", concise_prompt)
    print("Full messages:", json.dumps(msgs, indent=2, ensure_ascii=False))
    print("================================")
    # ========= END DEBUG =============

    r = requests.post(endpoint, headers=headers, json=data)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Azure GPT API error: {e}")
        return "Sorry, the assistant is currently unavailable."
    resp = r.json()
    logging.debug(f"Azure GPT raw response: {resp}")
    reply = resp['choices'][0]['message']['content']
    logging.info("Received reply from GPT.")
    return reply

# ─── Eleven Labs TTS ──────────────────────────────────────────────────

def elevenlabs_tts(text, out_path):
    import requests
    import os

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.getenv('ELEVEN_VOICE_ID')}"
    headers = {
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.7}
    }
    r = requests.post(url, json=data, headers=headers)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

# ─── Conversation End Detection ──────────────────────────────────
def is_convo_end(user_text, bot_text):
    end_phrases = ["bye", "nothing", "no", "that's all", "thank you"]
    text = (user_text + " " + bot_text).lower()
    result = any(kw in text for kw in end_phrases)
    logging.info(f"Conversation end detected? {result}")
    return result
