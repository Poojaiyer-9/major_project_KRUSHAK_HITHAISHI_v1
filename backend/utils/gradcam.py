import base64
import io

import cv2
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None


def get_gradcam(model, image_array, layer_name):
    """True Grad-CAM for a Keras model (used during development / evaluation).

    `model` may be a Keras Model whose first input is the 512x512x3 image and
    whose final output is the disease softmax. Returns a base64 JPEG overlay.
    """
    image = image_array.astype("float32") / 255.0
    image_colored = cv2.cvtColor(np.uint8(255 * image), cv2.COLOR_RGB2BGR)

    if tf is None or model is None:
        h, w = image_array.shape[:2] if image_array is not None else (224, 224)
        heatmap_colored = cv2.applyColorMap(
            np.zeros((h, w, 3), dtype=np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image_colored, 0.6, heatmap_colored, 0.4, 0)
        return _encode(overlay)

    img_batch = np.expand_dims(image, axis=0)
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        if isinstance(predictions, list):
            predictions = predictions[0]
        class_idx = int(np.argmax(predictions[0]))
        loss = predictions[0][class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy().squeeze(), 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    img_h, img_w = image_colored.shape[:2]
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_colored, 0.6, heatmap_colored, 0.4, 0)
    return _encode(overlay)


def get_gradcam_tflite(predict_proba_fn, image_array, disease_idx, grid=8):
    """Gradient-free, TFLite-compatible saliency heatmap via patch occlusion.

    `predict_proba_fn` takes an image array and returns a softmax vector over
    disease classes. We occlude a grid of patches with the mean image colour and
    measure the drop in probability for `disease_idx`, building a CAM-style map.
    """
    h, w = image_array.shape[:2]
    base_proba = float(predict_proba_fn(image_array)[disease_idx])
    occlusion_color = np.uint8(np.mean(image_array, axis=(0, 1)))
    patch_h, patch_w = h // grid, w // grid
    heatmap = np.zeros((grid, grid), dtype="float32")

    for i in range(grid):
        for j in range(grid):
            occluded = image_array.copy()
            occluded[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w] = occlusion_color
            proba = float(predict_proba_fn(occluded)[disease_idx])
            heatmap[i, j] = max(base_proba - proba, 0.0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_colored = cv2.cvtColor(np.uint8(image_array), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_colored, 0.6, heatmap_colored, 0.4, 0)
    return _encode(overlay)


def _encode(overlay_bgr):
    pil_image = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
