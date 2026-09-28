import os

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "krushak.tflite")
DISEASE_LABELS_PATH = os.path.join(MODELS_DIR, "disease_labels.txt")
SEVERITY_LABELS_PATH = os.path.join(MODELS_DIR, "severity_labels.txt")

SEVERITY_FALLBACK = ["LOW", "MEDIUM", "HIGH"]


def _load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


DISEASE_CLASSES = _load_lines(DISEASE_LABELS_PATH)
SEVERITY_CLASSES = _load_lines(SEVERITY_LABELS_PATH) or SEVERITY_FALLBACK

_interpreter = None


def get_interpreter():
    """Load (and cache) the TFLite interpreter. Returns None if the model is absent."""
    global _interpreter
    if tf is None:
        return None
    if _interpreter is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


def model_available() -> bool:
    return get_interpreter() is not None


def _find_input(details, shape_signature):
    """Find input tensor matching a shape signature.
    shape_signature[-1] must match exactly; other dims can be None (wildcard).
    """
    for d in details:
        if len(d["shape"]) == len(shape_signature) and d["shape"][-1] == shape_signature[-1]:
            if all(s == sh or sh is None for s, sh in zip(d["shape"][1:-1], shape_signature[1:-1])):
                return d
    # fallback: match by total rank and last dim only
    for d in details:
        if len(d["shape"]) == len(shape_signature) and d["shape"][-1] == shape_signature[-1]:
            return d
    raise ValueError(f"Could not locate model input tensor matching last_dim={shape_signature[-1]}")


def _get_image_size(interpreter) -> int:
    """Auto-detect the image H/W from the TFLite interpreter's input tensors."""
    for d in interpreter.get_input_details():
        if len(d["shape"]) == 4 and d["shape"][-1] == 3:
            return int(d["shape"][1])  # H == W for square images
    return 224  # default fallback


def _quantize(tensor_detail, array):
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    qp = tensor_detail.get("quantization_parameters", {})
    scale = qp.get("scales")
    zero_point = qp.get("zero_points")
    if scale and zero_point and scale[0] != 0:
        return (array / scale[0] + zero_point[0]).astype(tensor_detail["dtype"])
    return array.astype(tensor_detail["dtype"])


def _preprocess_image(image_array):
    img = np.asarray(image_array, dtype="float32") / 255.0
    return np.expand_dims(img, axis=0)


def _dequantize(tensor_detail, array):
    qp = tensor_detail.get("quantization_parameters", {})
    scale = qp.get("scales")
    zero_point = qp.get("zero_points")
    if scale and zero_point and scale[0] != 0:
        return (array.astype("float32") - zero_point[0]) * scale[0]
    return array.astype("float32")


def _aux_vector(weather, crop_stage):
    stage_value = {"seedling": 0.0, "vegetative": 1.0, "flowering": 2.0}.get(crop_stage, 1.0)
    return np.array([[
        float(weather.get("temperature_2m", 28.0)),
        float(weather.get("relative_humidity_2m", 65.0)),
        float(weather.get("precipitation", 0.0)),
        stage_value,
    ]], dtype="float32")


def _set_inputs(interpreter, image_array, weather, crop_stage):
    details = interpreter.get_input_details()
    img_size = _get_image_size(interpreter)   # auto-detect from model (224 or 512)
    image_detail = _find_input(details, (1, img_size, img_size, 3))
    aux_detail = _find_input(details, (1, 4))
    # Resize image to match what the model expects
    if image_array.shape[0] != img_size or image_array.shape[1] != img_size:
        from PIL import Image as _PIL
        pil = _PIL.fromarray(image_array.astype("uint8")).resize((img_size, img_size))
        image_array = np.array(pil)
    image_data = _quantize(image_detail, _preprocess_image(image_array))
    aux_data = _quantize(aux_detail, _aux_vector(weather, crop_stage))
    interpreter.set_tensor(image_detail["index"], image_data)
    interpreter.set_tensor(aux_detail["index"], aux_data)


def _disease_output_detail(interpreter):
    for out in interpreter.get_output_details():
        last = out["shape"][-1]
        if last == len(DISEASE_CLASSES) or (len(DISEASE_CLASSES) == 0 and last == 38):
            return out
    return interpreter.get_output_details()[0]


def _severity_output_detail(interpreter):
    for out in interpreter.get_output_details():
        last = out["shape"][-1]
        if last == len(SEVERITY_CLASSES) or (len(SEVERITY_CLASSES) == 3 and last == 3):
            return out
    return interpreter.get_output_details()[-1]


def predict_proba(image_array, weather, crop_stage="vegetative"):
    """Return softmax probabilities over disease classes for a single image."""
    interpreter = get_interpreter()
    if interpreter is None:
        return None
    _set_inputs(interpreter, image_array, weather, crop_stage)
    interpreter.invoke()
    detail = _disease_output_detail(interpreter)
    raw = _dequantize(detail, interpreter.get_tensor(detail["index"]))
    exp = np.exp(raw - np.max(raw))
    proba = exp / np.sum(exp)
    # Squeeze batch dim: always return shape [num_classes] not [1, num_classes]
    return proba.squeeze()


def predict(image_array, weather, crop_stage="vegetative"):
    """Run full inference. Returns dict with disease, severity, confidence (or None)."""
    interpreter = get_interpreter()
    if interpreter is None:
        return None
    _set_inputs(interpreter, image_array, weather, crop_stage)
    interpreter.invoke()

    disease_detail = _disease_output_detail(interpreter)
    severity_detail = _severity_output_detail(interpreter)
    disease_raw = _dequantize(disease_detail, interpreter.get_tensor(disease_detail["index"]))[0]
    severity_raw = _dequantize(severity_detail, interpreter.get_tensor(severity_detail["index"]))[0]

    disease_idx = int(np.argmax(disease_raw))
    severity_idx = int(np.argmax(severity_raw))
    disease_name = DISEASE_CLASSES[disease_idx] if DISEASE_CLASSES else f"class_{disease_idx}"
    severity_name = SEVERITY_CLASSES[severity_idx] if SEVERITY_CLASSES else "MEDIUM"
    return {
        "disease_name": disease_name,
        "disease_index": disease_idx,
        "severity": severity_name,
        "confidence": float(disease_raw[disease_idx]),
    }
