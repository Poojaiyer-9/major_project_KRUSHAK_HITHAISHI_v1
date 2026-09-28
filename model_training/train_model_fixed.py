# Fixed train_model.py — with these improvements:
# 1. Uses imagenet pretrained weights (much better convergence)
# 2. Exports disease_labels.txt and severity_labels.txt alongside model
# 3. Reduced image size to 224x224 to speed up local training
# 4. Configurable image_size parameter

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


def build_model(num_disease_classes: int = 38, num_severity_classes: int = 3, image_size: int = 224):
    image_input = keras.Input(shape=(image_size, image_size, 3), name="image")
    weather_input = keras.Input(shape=(4,), name="weather")

    base_model = keras.applications.MobileNetV3Small(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        weights="imagenet",  # FIX: use pretrained weights for faster convergence
    )
    base_model.trainable = True  # fine-tune

    image_features = base_model(image_input, training=False)
    image_features = layers.GlobalAveragePooling2D()(image_features)
    image_features = layers.Dense(64, activation="relu")(image_features)

    weather_features = layers.Dense(64, activation="relu")(weather_input)
    merged = layers.Concatenate()([image_features, weather_features])
    merged = layers.Dense(256, activation="relu")(merged)
    merged = layers.Dropout(0.3)(merged)

    disease_output = layers.Dense(num_disease_classes, activation="softmax", name="disease")(merged)
    severity_output = layers.Dense(num_severity_classes, activation="softmax", name="severity")(merged)

    model = keras.Model(inputs=[image_input, weather_input], outputs=[disease_output, severity_output])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={"disease": "categorical_crossentropy", "severity": "categorical_crossentropy"},
        metrics={"disease": "accuracy", "severity": "accuracy"},
    )
    return model


def _encode(labels, severities):
    disease_classes = sorted(set(labels))
    severity_classes = ["LOW", "MEDIUM", "HIGH"]
    disease_idx = {c: i for i, c in enumerate(disease_classes)}
    severity_idx = {c: i for i, c in enumerate(severity_classes)}
    y_disease = to_categorical([disease_idx[l] for l in labels], num_classes=len(disease_classes))
    y_severity = to_categorical([severity_idx[s] for s in severities], num_classes=3)
    return y_disease, y_severity, len(disease_classes), disease_classes


def train(
    dataset_path: str = "./dataset.npz",
    model_out: str = "./model.keras",
    labels_out: str = "./",
    image_size: int = 224,
):
    data = np.load(dataset_path, allow_pickle=True)
    images = data["images"].astype("float32") / 255.0
    weather = data["weather"].astype("float32")
    y_disease, y_severity, n_classes, disease_classes = _encode(data["labels"], data["severities"])

    print(f"Dataset: {len(images)} samples, {n_classes} disease classes")

    # Resize images if they're not the target size already
    if images.shape[1] != image_size:
        from PIL import Image as PILImage
        print(f"Resizing images from {images.shape[1]}x{images.shape[2]} to {image_size}x{image_size}...")
        resized = []
        for i, img in enumerate(images):
            pil = PILImage.fromarray((img * 255).astype("uint8")).resize((image_size, image_size))
            resized.append(np.array(pil).astype("float32") / 255.0)
            if i % 1000 == 0:
                print(f"  Resized {i}/{len(images)}")
        images = np.array(resized)

    (
        X_img_tr, X_img_te, X_w_tr, X_w_te, yd_tr, yd_te, ys_tr, ys_te,
    ) = train_test_split(
        images, weather, y_disease, y_severity, test_size=0.2, random_state=42
    )
    X_img_tr, X_img_val, X_w_tr, X_w_val, yd_tr, yd_val, ys_tr, ys_val = train_test_split(
        X_img_tr, X_w_tr, yd_tr, ys_tr, test_size=0.125, random_state=42
    )

    model = build_model(num_disease_classes=n_classes, image_size=image_size)
    model.summary()

    model.fit(
        {"image": X_img_tr, "weather": X_w_tr},
        {"disease": yd_tr, "severity": ys_tr},
        validation_data=({"image": X_img_val, "weather": X_w_val}, {"disease": yd_val, "severity": ys_val}),
        epochs=20,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ],
    )
    model.save(model_out)
    print(f"Saved trained model to {model_out} ({n_classes} disease classes)")

    # Export label files
    disease_labels_path = os.path.join(labels_out, "disease_labels.txt")
    severity_labels_path = os.path.join(labels_out, "severity_labels.txt")
    with open(disease_labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(disease_classes))
    with open(severity_labels_path, "w", encoding="utf-8") as f:
        f.write("LOW\nMEDIUM\nHIGH")
    print(f"Saved {disease_labels_path}")
    print(f"Saved {severity_labels_path}")


if __name__ == "__main__":
    train()
