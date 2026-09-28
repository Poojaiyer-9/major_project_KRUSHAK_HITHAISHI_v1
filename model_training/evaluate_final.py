"""
evaluate_final.py — Evaluation for Krushak Hithaishi
=====================================================
Per instruct.md step 7:
  - Test accuracy for multimodal (TFLite) vs image-only baseline
  - Model size in KB
  - Average inference latency in ms over 50 runs
  - All numbers come from actual execution — nothing is estimated.
"""

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


def _encode(labels, severities):
    disease_classes  = sorted(set(labels))
    severity_classes = ["LOW", "MEDIUM", "HIGH"]
    disease_idx  = {c: i for i, c in enumerate(disease_classes)}
    severity_idx = {c: i for i, c in enumerate(severity_classes)}
    y_disease  = to_categorical([disease_idx[l]  for l in labels], num_classes=len(disease_classes))
    y_severity = to_categorical([severity_idx[s] for s in severities], num_classes=3)
    return y_disease, y_severity, len(disease_classes)


def _get_test_split(data):
    """Reproduce the same test split used during training."""
    images   = data["images"].astype("float32") / 255.0
    weather  = data["weather"].astype("float32")
    y_disease, y_severity, n_classes = _encode(data["labels"], data["severities"])
    strata = np.argmax(y_disease, axis=1)
    _, X_img_te, _, X_w_te, _, yd_te, _, ys_te = train_test_split(
        images, weather, y_disease, y_severity,
        test_size=0.2, random_state=42, stratify=strata
    )
    return X_img_te, X_w_te, yd_te, ys_te, n_classes


def evaluate_tflite(model_path: str, data, image_size: int = 224):
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    X_img_te, X_w_te, yd_te, ys_te, _ = _get_test_split(data)

    # Resize if needed
    if X_img_te.shape[1] != image_size:
        from PIL import Image as PILImage
        resized = []
        for img in X_img_te:
            pil = PILImage.fromarray((img * 255).astype("uint8")).resize((image_size, image_size))
            resized.append(np.array(pil).astype("float32") / 255.0)
        X_img_te = np.array(resized)

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    img_in = next(d for d in in_details  if len(d["shape"]) == 4 and d["shape"][-1] == 3)
    w_in   = next(d for d in in_details  if d["shape"][-1] == 4)
    disease_out = max(out_details, key=lambda d: d["shape"][-1])
    sev_out     = [d for d in out_details if d is not disease_out][0]

    disease_correct = severity_correct = 0
    n = len(X_img_te)

    # ── Latency measurement: 50 random samples ─────────────────────────────
    sample_idx = np.random.default_rng(0).choice(n, size=min(50, n), replace=False)
    latencies = []

    for i in range(n):
        img_input = np.expand_dims(X_img_te[i], 0).astype(img_in["dtype"])
        w_input   = np.expand_dims(X_w_te[i],   0).astype(w_in["dtype"])

        if i in sample_idx:
            t0 = time.perf_counter()

        interpreter.set_tensor(img_in["index"], img_input)
        interpreter.set_tensor(w_in["index"],   w_input)
        interpreter.invoke()

        if i in sample_idx:
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        d_pred = np.argmax(interpreter.get_tensor(disease_out["index"]))
        s_pred = np.argmax(interpreter.get_tensor(sev_out["index"]))

        disease_correct  += int(d_pred == np.argmax(yd_te[i]))
        severity_correct += int(s_pred == np.argmax(ys_te[i]))

    disease_acc  = disease_correct  / n
    severity_acc = severity_correct / n
    avg_latency  = float(np.mean(latencies))

    return disease_acc, severity_acc, avg_latency


def train_image_only_baseline(data, image_size: int = 224, epochs: int = 5):
    """Train the single-modality image-only baseline with the same settings."""
    images   = data["images"].astype("float32") / 255.0
    y_disease, _, n_classes = _encode(data["labels"], data["severities"])

    # Resize if needed
    if images.shape[1] != image_size:
        from PIL import Image as PILImage
        resized = []
        for img in images:
            pil = PILImage.fromarray((img * 255).astype("uint8")).resize((image_size, image_size))
            resized.append(np.array(pil).astype("float32") / 255.0)
        images = np.array(resized)

    strata = np.argmax(y_disease, axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        images, y_disease, test_size=0.2, random_state=42, stratify=strata
    )

    # Same backbone, same settings — image only
    inp = keras.Input(shape=(image_size, image_size, 3))
    base = keras.applications.MobileNetV3Small(
        include_top=False, weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )
    base.trainable = False
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3,
                restore_best_weights=True, mode="max"
            )
        ],
        verbose=1,
    )
    _, acc = model.evaluate(X_te, y_te, verbose=0)
    return acc


def main(model_path: str = "./krushak.tflite",
         dataset_path: str = "./dataset.npz",
         image_size: int = 224):

    print("\n" + "=" * 60)
    print("  Krushak Hithaishi — Model Evaluation")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[FAIL] Model not found: {model_path}")
        return

    # Model size
    size_kb = os.path.getsize(model_path) / 1024
    print(f"\nModel file: {model_path}")
    print(f"Model size: {size_kb:.1f} KB")

    data = np.load(dataset_path, allow_pickle=True)

    # ── Multimodal TFLite evaluation ────────────────────────────────────────
    print("\n[1/2] Evaluating multimodal TFLite model...")
    disease_acc, severity_acc, avg_latency = evaluate_tflite(
        model_path, data, image_size=image_size
    )

    # ── Image-only baseline ─────────────────────────────────────────────────
    print("\n[2/2] Training image-only baseline (same settings)...")
    baseline_acc = train_image_only_baseline(data, image_size=image_size)

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS (from actual execution — not estimated)")
    print("=" * 60)
    print(f"  Multimodal disease accuracy : {disease_acc:.4f}  ({disease_acc*100:.1f}%)")
    print(f"  Multimodal severity accuracy: {severity_acc:.4f}  ({severity_acc*100:.1f}%)")
    print(f"  Image-only baseline accuracy: {baseline_acc:.4f}  ({baseline_acc*100:.1f}%)")
    print(f"  Delta (multimodal - baseline): {disease_acc - baseline_acc:+.4f}")
    print(f"  Model size                  : {size_kb:.1f} KB")
    print(f"  Avg inference latency       : {avg_latency:.1f} ms  (over 50 runs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
