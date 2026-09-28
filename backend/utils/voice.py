import os
from datetime import datetime, timedelta

try:
    from gtts import gTTS
except Exception:  # pragma: no cover - optional dependency
    gTTS = None

VOICE_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "voice")
os.makedirs(VOICE_DIR, exist_ok=True)


def text_to_voice(text: str, lang_code: str = "kn"):
    if gTTS is None:
        return None
    try:
        import uuid

        filename = f"{uuid.uuid4().hex}.mp3"
        path = os.path.join(VOICE_DIR, filename)
        gTTS(text=text, lang=lang_code).save(path)
        return filename
    except Exception:
        return None


def cleanup_old_files(max_age_hours: int = 1):
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    for name in os.listdir(VOICE_DIR):
        if not name.endswith(".mp3"):
            continue
        path = os.path.join(VOICE_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff.timestamp():
                os.remove(path)
        except OSError:
            continue
