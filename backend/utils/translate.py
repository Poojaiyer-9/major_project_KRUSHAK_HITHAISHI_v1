try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

MODEL_MAP = {
    "kn": "ai4bharat/indictrans2-en-kn-1B",
    "hi": "ai4bharat/indictrans2-en-hi-1B",
    "te": "ai4bharat/indictrans2-en-te-1B",
}

_LANG_TOKEN = {"kn": "<2kn>", "hi": "<2hi>", "te": "<2te>"}

_CACHE = {}


def _load_model(target_lang: str):
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None
    if target_lang in _CACHE:
        return _CACHE[target_lang]
    model_name = MODEL_MAP.get(target_lang)
    if model_name is None:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _CACHE[target_lang] = (tokenizer, model)
    return _CACHE[target_lang]


def translate_text(text: str, target_lang: str = "kn"):
    try:
        loaded = _load_model(target_lang)
        if loaded is None:
            return text
        tokenizer, model = loaded
        prompt = f"{_LANG_TOKEN.get(target_lang, '')} {text}".strip()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=128)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception:
        return text
