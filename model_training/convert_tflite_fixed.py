# Fixed convert_tflite.py — fixes:
# 1. Representative dataset generator yields dict (required for multi-input models)
# 2. Handles both 512x512 and 224x224 image sizes
# 3. Falls back to float16 quantization if INT8 fails

import os
import numpy as np
import tensorflow as tf


def representative_dataset_gen(dataset_path: str = "./dataset.npz", num_samples: int = 200, image_size: int = 224):
    data = np.load(dataset_path, allow_pickle=True)
    images = data["images"].astype("float32") / 255.0
    weather = data["weather"].astype("float32")

    # Resize if needed
    if images.shape[1] != image_size:
        from PIL import Image as PILImage
        print(f"  Resizing representative set from {images.shape[1]} to {image_size}...")
        resized = []
        for img in images[:num_samples]:
            pil = PILImage.fromarray((img * 255).astype("uint8")).resize((image_size, image_size))
            resized.append(np.array(pil).astype("float32") / 255.0)
        images = np.array(resized)
    else:
        images = images[:num_samples]
        weather = weather[:num_samples]

    idx = np.linspace(0, len(images) - 1, min(num_samples, len(images))).astype(int)
    for i in idx:
        # FIX: yield as dict (required for named multi-input models)
        yield {
            "image": np.expand_dims(images[i], axis=0),
            "weather": np.expand_dims(weather[i], axis=0),
        }


def convert_model(
    model_path: str = "./model.keras",
    dataset_path: str = "./dataset.npz",
    output_path: str = "./krushak.tflite",
    image_size: int = 224,
    quantization: str = "int8",  # "int8" or "float16"
):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "int8":
        print("Applying INT8 post-training quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_gen(
            dataset_path, num_samples=200, image_size=image_size
        )
        # Use TFLITE_BUILTINS so it doesn't fail on unsupported INT8 ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        # Keep inputs/outputs as float for easier inference
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[WARN] INT8 conversion failed ({e}), falling back to float16...")
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.target_spec.supported_types = [tf.float16]
        tflite_model = converter2.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {output_path} ({size_kb:.1f} KB)")
    return output_path


if __name__ == "__main__":
    convert_model()
