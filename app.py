"""
app.py
------
Streamlit frontend for the Face Detection System.

This app provides a modern, clean UI that allows users to:
    1. Upload an image and detect faces in it.
    2. Choose between Deep Learning (YuNet - Recommended) and Classic Haar Cascade models.
    3. Fine-tune detection sensitivity and parameters (Confidence threshold, Scale factor, Min Neighbors, Contrast enhancement).

All actual face-detection logic lives in `train_model.py`.

Run with:
    streamlit run app.py
"""

import os
import importlib
import streamlit as st

# Import numeric and OpenCV libraries with helpful error messages
try:
    import numpy as np
except Exception as exc:
    msg = (
        "Failed to import `numpy`. Reinstall NumPy before running the app:\n\n"
        "    pip install --upgrade --force-reinstall numpy"
    )
    try:
        st.error(msg)
        st.stop()
    except Exception:
        print(msg)
    raise

try:
    import cv2
except Exception as exc:
    msg = (
        "Failed to import `cv2` (OpenCV). Reinstall OpenCV:\n\n"
        "    pip install opencv-python"
    )
    try:
        st.error(msg)
        st.stop()
    except Exception:
        print(msg)
    raise

# Import backend module and reload to ensure fresh execution in Streamlit
import train_model
importlib.reload(train_model)

from train_model import (
    detect_faces,
    load_face_cascade,
    get_face_cascade,
    BASE_DIR,
    save_uploaded_image,
    save_output_image,
    CASCADE_ENV_VAR,
)


@st.cache_resource
def get_cascade_cached():
    """Load and cache the Haar Cascade face detector."""
    return get_face_cascade()


LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Detection System",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------------------------
def apply_custom_styles():
    """Inject custom CSS to give the app a modern, professional look."""
    st.markdown(
        """
        <style>
        .stApp, .stApp * {
            background-color: white !important;
            color: black !important;
        }

        /* Buttons */
        div.stButton > button {
            color: white !important;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            border: none;
            background: linear-gradient(90deg, #4f46e5, #06b6d4);
            transition: 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0px 4px 14px rgba(79, 70, 229, 0.35);
        }

        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: black;
            margin-bottom: 0px;
        }

        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            color: #555555;
            margin-top: 0px;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }

        .footer {
            text-align: center;
            color: #777777;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS & INFO
# ---------------------------------------------------------------------------
def render_sidebar():
    """
    Render sidebar with logo, detection settings, model selection, and project info.
    
    Returns:
        dict: Configured detection options.
    """
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

        st.header("⚙️ Detection Settings")
        
        model_choice = st.radio(
            "Select Detection Engine:",
            (
                "🤖 Deep Learning (YuNet - Highly Accurate)",
                "🧠 Haar Cascade (Classic OpenCV)"
            ),
            index=0,
            help="YuNet Deep Learning model is highly accurate on complex group photos, angled faces, and low light."
        )

        model_type = "yunet" if "YuNet" in model_choice else "haar"

        options = {"model_type": model_type}

        if model_type == "yunet":
            st.subheader("🎯 YuNet Parameters")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.10,
                max_value=1.00,
                value=0.65,
                step=0.05,
                help="Lower threshold detects subtle/small faces; higher threshold eliminates weak detections."
            )
            options["confidence_threshold"] = confidence_threshold
        else:
            st.subheader("🛠️ Haar Cascade Parameters")
            scale_factor = st.slider(
                "Scale Factor",
                min_value=1.02,
                max_value=1.30,
                value=1.05,
                step=0.01,
                help="Image scale reduction step size. Lower values (e.g. 1.05) increase accuracy for smaller faces."
            )
            min_neighbors = st.slider(
                "Min Neighbors",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Higher values reduce false positives; lower values detect more potential faces."
            )
            use_clahe = st.checkbox(
                "Apply Contrast Enhancement (CLAHE)",
                value=True,
                help="Improves face detection in shadows or uneven lighting conditions."
            )
            options["scale_factor"] = scale_factor
            options["min_neighbors"] = min_neighbors
            options["use_clahe"] = use_clahe

        st.divider()

        st.header("ℹ️ Project Information")
        st.markdown("**Face Detection System**")
        st.write(
            "Multi-model real-time face detection system powered by OpenCV Deep Learning (YuNet) "
            "and Haar Cascade classifiers."
        )

        st.subheader("🛠️ Technologies")
        st.markdown(
            """
            - 🐍 Python & Streamlit
            - 📷 OpenCV (cv2)
            - 🤖 YuNet Deep Learning (ONNX)
            - 🧠 Haar Cascade
            """
        )

        st.subheader("👨‍💻 Developer")
        st.write("Deepak Bhati")

        return options


# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
def render_header():
    """Render main title and subtitle."""
    st.markdown('<p class="main-title">Face Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">High-Accuracy AI Face Detection using OpenCV YuNet Deep Learning & Haar Cascade.</p>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# RESULTS DISPLAY
# ---------------------------------------------------------------------------
def render_results(face_count, options):
    """Display detection metrics and status messages."""
    st.markdown("### 📊 Detection Results")

    model_display_name = (
        "YuNet Deep Learning (ONNX)" if options["model_type"] == "yunet" else "Haar Cascade (Classic)"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Faces Detected", face_count)
    col2.metric("Processing Status", "Completed ✅")
    col3.metric("Detection Model", model_display_name)

    if face_count > 0:
        st.success(f"Face detection completed successfully! **{face_count} faces detected** using {model_display_name}.")
    else:
        st.warning("No faces detected with current settings. Try lowering confidence threshold or scale factor in the sidebar.")


# ---------------------------------------------------------------------------
# UPLOAD TAB
# ---------------------------------------------------------------------------
def render_upload_tab(options):
    """Render image upload and detection workflow."""
    st.markdown("#### 📤 Upload an image to detect faces")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.warning("Please upload an image to begin face detection.")
        return

    # Convert uploaded file into an OpenCV-compatible BGR image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_image is None:
        st.error("The uploaded file could not be read as a valid image.")
        return

    # Save original upload into uploads/
    save_uploaded_image(original_image, uploaded_file.name)

    col_original, col_detected = st.columns(2)

    with col_original:
        st.markdown("**Original Image**")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Run face detection with chosen settings
    if options["model_type"] == "yunet":
        processed_image, face_count = detect_faces(
            original_image,
            model_type="yunet",
            confidence_threshold=options.get("confidence_threshold", 0.65)
        )
    else:
        processed_image, face_count = detect_faces(
            original_image,
            model_type="haar",
            scale_factor=options.get("scale_factor", 1.05),
            min_neighbors=options.get("min_neighbors", 4),
            use_clahe=options.get("use_clahe", True)
        )

    # Save output into outputs/
    name, ext = os.path.splitext(uploaded_file.name)
    output_filename = f"{name}_detected{ext}"
    save_output_image(processed_image, output_filename)

    with col_detected:
        st.markdown("**Detected Faces**")
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    render_results(face_count, options)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
def render_footer():
    """Render footer."""
    st.markdown(
        '<p class="footer">Built with ❤️ using Streamlit + OpenCV (YuNet & Haar Cascade)</p>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# MAIN APP ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit application entry point."""
    apply_custom_styles()
    options = render_sidebar()
    render_header()
    render_upload_tab(options)
    render_footer()


if __name__ == "__main__":
    main()
