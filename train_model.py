import os
import json
import zipfile
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. CONFIGURATION
# ==========================================

ZIP_PATH = "cat breeds(1).zip"
DATASET_PATH = "cat-breeds/cat-breeds"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

MODEL_PATH = "cat_breed_model.keras"
CLASS_PATH = "class_names.json"

print("====================================")
print(" CAT BREED CLASSIFICATION")
print("====================================")


# ==========================================
# 2. EXTRACT DATASET
# ==========================================

if not os.path.isdir(DATASET_PATH):

    print("\nExtracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset extracted successfully!")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nNumber of cat breeds:", num_classes)

print("\nAvailable breeds:")

for breed in class_names:
    print("-", breed.replace("_", " ").title())


# Save the exact class ordering used during training.
with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)


# ==========================================
# 4. OPTIMIZE DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])


# ==========================================
# 6. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially.
base_model.trainable = False


# ==========================================
# 7. BUILD CLASSIFICATION MODEL
# ==========================================

inputs = layers.Input(
    shape=(160, 160, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 expects inputs scaled to [-1, 1].
x = layers.Lambda(
    preprocess_input,
    name="mobilenet_preprocessing"
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\n====================================")
print(" TRAINING COMPLETED")
print("====================================")

print(f"Validation Loss: {loss:.4f}")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

print("\nModel saved as:", MODEL_PATH)
print("Class names saved as:", CLASS_PATH)

print("\nYou can now run the Streamlit application.")