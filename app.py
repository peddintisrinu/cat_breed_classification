
import os
import json
import hashlib

import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cat Breed Classification",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DAY / NIGHT THEME
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.title("⚙️ Settings")

    st.toggle(
        "🌙 Night Mode",
        key="dark_mode"
    )

    st.divider()

    st.subheader("About the Application")

    st.write(
        "This application uses a deep-learning image "
        "classification model to predict the possible "
        "breed of a cat."
    )

    st.caption("Technology: TensorFlow + Keras + Streamlit")


dark = st.session_state.dark_mode

if dark:
    bg = "#101827"
    card = "#1c293b"
    text = "#f1f5f9"
    muted = "#b6c2d2"
    border = "#35445b"
    accent = "#60a5fa"
    result_bg = "#20364b"
else:
    bg = "#f3f6fb"
    card = "#ffffff"
    text = "#172b4d"
    muted = "#5b6575"
    border = "#dce4ef"
    accent = "#2563eb"
    result_bg = "#e8f5ee"


st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    [data-testid="stSidebar"] {{
        background-color: {card};
    }}

    [data-testid="stHeader"] {{
        background-color: {bg};
    }}

    h1, h2, h3, h4, p, label, li {{
        color: {text};
    }}

    .hero {{
        padding: 28px 18px;
        border-radius: 18px;
        text-align: center;
        background: {card};
        border: 1px solid {border};
        margin-bottom: 22px;
    }}

    .hero h1 {{
        color: {accent};
        font-size: 38px;
        margin-bottom: 10px;
    }}

    .hero p {{
        color: {muted};
        font-size: 17px;
    }}

    .info-card {{
        background: {card};
        border: 1px solid {border};
        padding: 20px;
        border-radius: 14px;
        margin: 10px 0;
    }}

    .result-card {{
        background: {result_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 15px 0;
    }}

    .breed {{
        font-size: 30px;
        font-weight: 800;
        color: {accent};
    }}

    .confidence {{
        font-size: 22px;
        font-weight: bold;
        color: {text};
    }}

    .muted {{
        color: {muted};
    }}

    div[data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        padding: 15px;
        border-radius: 12px;
    }}

    .footer {{
        text-align: center;
        padding: 24px;
        color: {muted};
        font-size: 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "cat_breed_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# =========================================================
# 4. LOAD CLASS NAMES
# =========================================================

@st.cache_data
def load_class_names(path):
    with open(path, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, dict):
        try:
            names = [
                value
                for key, value in sorted(
                    names.items(),
                    key=lambda item: int(item[0])
                )
            ]
        except ValueError:
            names = list(names.values())

    if not isinstance(names, list) or not names:
        raise ValueError(
            "class_names.json must contain a non-empty "
            "list or an index-to-class dictionary."
        )

    return names


# =========================================================
# 5. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
        safe_mode=False
    )


if not os.path.exists(MODEL_PATH):
    st.error("❌ cat_breed_model.keras was not found.")
    st.stop()

if not os.path.exists(CLASS_NAMES_PATH):
    st.error("❌ class_names.json was not found.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

except Exception as error:
    st.error("Unable to load the model or class names.")

    with st.expander("Technical details"):
        st.exception(error)

    st.stop()


# =========================================================
# 6. APPLICATION HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🐱 Cat Breed Classification</h1>
        <p>
            Discover the possible breed of your cat using
            deep learning and image recognition.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7. MODEL INFORMATION
# =========================================================

with st.expander("🧠 Learn About the Model", expanded=False):

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("### Deep Learning")

        st.write(
            "Deep learning uses neural networks to learn patterns "
            "from training images. An image classifier learns "
            "visual features that may help distinguish cat breeds."
        )

        st.markdown("### Image Processing")

        st.write(
            "The uploaded image is converted to RGB and resized "
            "to the dimensions expected by the trained model."
        )

    with info2:
        st.markdown("### Classification")

        st.write(
            "The model produces a score for each breed in its "
            "training classes. The class with the highest score "
            "is selected as the leading prediction."
        )

        st.markdown("### Technologies Used")

        st.write(
            "- TensorFlow and Keras: model inference\n"
            "- MobileNetV2 preprocessing: if used during training\n"
            "- NumPy: numerical processing\n"
            "- Pillow: image loading\n"
            "- Streamlit: web interface"
        )

    st.warning(
        "The model can only predict breeds represented in its "
        "training classes. Prediction scores are not guarantees "
        "of correctness, and performance depends on training data."
    )


# =========================================================
# 8. IMAGE PREPROCESSING
# =========================================================

def prepare_image(image, classification_model):

    input_shape = classification_model.input_shape

    if isinstance(input_shape, list):
        raise ValueError(
            "This app supports models with one input tensor."
        )

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected model input shape: {input_shape}"
        )

    height = input_shape[1] or 160
    width = input_shape[2] or 160

    if input_shape[3] != 3:
        raise ValueError(
            "The model must accept three-channel RGB images."
        )

    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # This code assumes preprocessing is already inside
    # the saved model as the MobileNetV2 Lambda layer.
    # Do not apply preprocess_input a second time.

    return image_array


# =========================================================
# 9. PREDICTION FUNCTION
# =========================================================

def predict_breed(image, classification_model, labels):

    image_array = prepare_image(
        image,
        classification_model
    )

    raw_predictions = classification_model.predict(
        image_array,
        verbose=0
    )

    raw_predictions = np.asarray(raw_predictions)

    if raw_predictions.ndim != 2:
        raise ValueError(
            "The model must return classification scores "
            "with shape (batch_size, number_of_classes)."
        )

    scores = raw_predictions[0]

    if len(scores) != len(labels):
        raise ValueError(
            f"Model output has {len(scores)} scores, but "
            f"class_names.json contains {len(labels)} labels. "
            "Check the class names and their order."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid scores.")

    # Detect whether output already looks like probabilities.
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=1e-3)
    ):
        probabilities = scores
    else:
        probabilities = tf.nn.softmax(scores).numpy()

    predicted_index = int(np.argmax(probabilities))

    return predicted_index, probabilities


# =========================================================
# 10. UPLOAD IMAGE
# =========================================================

st.markdown("## 📸 Upload Your Cat Image")

st.write(
    "Choose a clear image of a cat. The application will "
    "automatically predict the possible breed after upload."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="cat_image"
)


# =========================================================
# 11. AUTOMATIC PREDICTION
# =========================================================

if uploaded_file is not None:

    # Identify the uploaded file so prediction can be cached.
    file_bytes = uploaded_file.getvalue()

    image_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:
        st.error("The uploaded file is not a valid image.")
        st.stop()

    image_column, result_column = st.columns(
        [1, 1],
        gap="large"
    )

    with image_column:

        st.markdown("### 🖼️ Uploaded Image")

        st.image(
            image,
            caption="Your uploaded cat image",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

    with result_column:

        st.markdown("### 🔍 Prediction Results")

        try:
            with st.spinner(
                "🧠 Analyzing your image..."
            ):

                # Predict once per unique image.
                @st.cache_data(show_spinner=False)
                def cached_prediction(file_hash, model_path):
                    del file_hash, model_path
                    return None

                # Run the actual prediction directly.
                # Streamlit reruns the script after upload.
                predicted_index, probabilities = predict_breed(
                    image,
                    model,
                    class_names
                )

                predicted_breed = class_names[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index] * 100
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="breed">🐱 {predicted_breed}</div>
                    <br>
                    <div class="confidence">
                        {confidence:.2f}% model score
                    </div>
                    <p class="muted">
                        Highest-scoring class for this image
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(
                    np.clip(confidence / 100, 0, 1)
                )
            )

            st.caption(
                "This score is the model's output for the "
                "predicted class, not a verified probability "
                "that the breed identification is correct."
            )

        except Exception as error:

            st.error("Prediction could not be completed.")

            with st.expander("Technical details"):
                st.exception(error)

            st.stop()


    # =====================================================
    # 12. TOP PREDICTIONS
    # =====================================================

    st.divider()

    st.markdown("## 📊 Top Breed Predictions")

    st.write(
        "These are the highest-scoring classes returned by "
        "the model for the uploaded image."
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:min(3, len(class_names))]

    top_columns = st.columns(
        len(top_indices)
    )

    for rank, (column, index) in enumerate(
        zip(top_columns, top_indices),
        start=1
    ):

        breed = class_names[int(index)]

        score = float(
            probabilities[index] * 100
        )

        with column:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>#{rank}</h3>
                    <h4>{breed}</h4>
                    <p><strong>{score:.2f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(
                    np.clip(score / 100, 0, 1)
                )
            )


    # =====================================================
    # 13. EXPLAIN THE RESULT
    # =====================================================

    st.divider()

    st.markdown("## 📝 Understanding Your Result")

    st.markdown(
        f"""
        <div class="info-card">
            <h3>Predicted breed: {predicted_breed}</h3>
            <p>
                The model assigned the highest score to
                <strong>{predicted_breed}</strong> among its
                available classification classes.
            </p>
            <p>
                Its output score is
                <strong>{confidence:.2f}%</strong>.
                This indicates how the model scored this class
                relative to the other available classes.
            </p>
            <p>
                The prediction may be affected by image quality,
                lighting, the cat's pose, background objects,
                similarities between breeds, and the diversity
                of the training dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if confidence >= 80:
        st.success(
            "The model gives a strong score to its leading class. "
            "This does not guarantee that the breed is correct."
        )

    elif confidence >= 50:
        st.info(
            "The model favors this class, but comparing the "
            "other predictions may be useful."
        )

    else:
        st.warning(
            "The model's leading score is relatively low. "
            "Try a clearer image with the cat fully visible."
        )

    st.markdown("### 💡 Tips for Better Predictions")

    st.markdown(
        """
        - Upload a clear, well-lit image.
        - Keep the cat's face and body visible.
        - Avoid blurry or very small images.
        - Try a different image if the result seems incorrect.
        - Remember that similar-looking breeds can be difficult
          for an image classifier to distinguish.
        """
    )

else:

    st.info(
        "👆 Upload a cat image above to automatically see "
        "the prediction and detailed results."
    )

    st.markdown("## ✨ What You Can Explore")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🐱 Breed Prediction</h3>
                <p>
                    Identify the highest-scoring breed
                    from the model's known classes.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Score Analysis</h3>
                <p>
                    Compare the leading predictions
                    and their model output scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>🧠 Deep Learning</h3>
                <p>
                    Explore how image classification
                    models learn visual patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 14. MODEL DETAILS
# =========================================================

with st.expander("🔬 Technical Model Details"):

    st.write("**Framework:** TensorFlow / Keras")

    st.write(
        "**Model input shape:**",
        model.input_shape
    )

    st.write(
        "**Model output shape:**",
        model.output_shape
    )

    st.write(
        "**Number of available classes:**",
        len(class_names)
    )

    st.write(
        "**TensorFlow version:**",
        tf.__version__
    )

    st.write(
        "**Image preprocessing:**",
        "RGB conversion and resizing to the model's input dimensions."
    )

    st.write(
        "**Architecture:**",
        "Check your training code to confirm whether the saved "
        "model uses MobileNetV2 or another architecture."
    )

    st.write(
        "**Important:** This application uses an existing trained "
        "model. Its accuracy must be measured using an independent "
        "test dataset; it cannot be determined from one prediction."
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🐾 Cat Breed Classification System<br>
        Built with Python, TensorFlow, Keras and Streamlit<br>
        AI predictions are estimates and may be incorrect.
    </div>
    """,
    unsafe_allow_html=True
)

