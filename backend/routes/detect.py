import io

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image

from routes.advisory import _normalize, get_treatment_protocol
from utils.gradcam import get_gradcam, get_gradcam_tflite
from utils.model import model_available, predict, predict_proba
from utils.translate import translate_text
from utils.voice import text_to_voice
from utils.weather import get_weather

router = APIRouter(prefix="", tags=["detect"])

DEMO_DISEASE = "Potato___Late_blight"


def _display_name(disease_name: str) -> str:
    return _normalize(disease_name).title()


def _build_advisory(disease_name, severity, protocol):
    name = _display_name(disease_name)
    return (
        f"{name} detected with {severity.lower()} severity. "
        f"Recommended treatment: {protocol['medicine_name']} at {protocol['dosage_per_acre']}. "
        f"Organic alternative: {protocol['organic_alternative']}. "
        f"{protocol['application_timing']}."
    )


@router.post("/detect")
async def detect(
    request: Request,
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    crop_stage: str = Form("vegetative"),
    language: str = Form("kn"),
):
    weather = await get_weather(lat, lon)
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(pil_image.resize((512, 512)))

    result = predict(image_array, weather, crop_stage) if model_available() else None
    demo = result is None
    if demo:
        result = {
            "disease_name": DEMO_DISEASE,
            "disease_index": 0,
            "severity": "HIGH",
            "confidence": 0.91,
        }

    protocol = get_treatment_protocol(result["disease_name"])
    advisory_text = _build_advisory(result["disease_name"], result["severity"], protocol)

    translated = translate_text(advisory_text, target_lang=language)
    voice_filename = text_to_voice(translated, lang_code=language)

    if demo:
        heatmap = get_gradcam(None, image_array, "conv2d")
    else:
        heatmap = get_gradcam_tflite(
            lambda img: predict_proba(img, weather, crop_stage),
            image_array,
            result["disease_index"],
        )

    base = str(request.base_url).rstrip("/")
    voice_file_url = f"{base}/voice/{voice_filename}" if voice_filename else None
    return {
        "disease_name": result["disease_name"],
        "disease_display": _display_name(result["disease_name"]),
        "confidence": result["confidence"],
        "severity": result["severity"],
        "heatmap_base64": heatmap,
        "advisory_translated": translated,
        "advisory_english": advisory_text,
        "treatment": protocol,
        "voice_file_url": voice_file_url,
        "weather": weather,
        "crop_stage": crop_stage,
        "demo": demo,
    }
