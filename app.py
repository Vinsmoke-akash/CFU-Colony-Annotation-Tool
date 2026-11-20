
import json
import hashlib
from pathlib import Path
from collections import Counter

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CFU Colony Annotation Tool",
    layout="wide"
)

st.title("CFU Colony Annotation Tool")

st.write(
    "Upload a Petri-dish image to detect bacterial colonies "
    "automatically using YOLOv8 or annotate them manually."
)


# --------------------------------------------------
# YOLO MODEL PATH
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "best (2).pt"
)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:

    st.header("Manual Annotation Settings")

    colony_class = st.text_input(
        "Colony class",
        value="B.subtilis"
    )

    stroke_width = st.slider(
        "Rectangle line width",
        min_value=1,
        max_value=8,
        value=3
    )

    stroke_color = st.color_picker(
        "Rectangle color",
        "#00FF00"
    )

    fill_color = "rgba(0, 255, 0, 0.15)"

    st.divider()

    st.header("YOLO Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.20,
        step=0.05
    )

    st.caption(
        "Lower confidence may detect more colonies, "
        "but may also produce false detections."
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload Petri-dish image",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)


if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    image_key = hashlib.md5(
        uploaded.getvalue()
    ).hexdigest()

    # Clear previous results when image changes
    if st.session_state.get("current_image_key") != image_key:

        st.session_state["current_image_key"] = image_key

        for key in [
            "yolo_image",
            "yolo_counts",
            "yolo_total",
            "yolo_image_key",
            "yolo_confidences",
            "yolo_labels",
            "yolo_classes",
            "yolo_debug"
        ]:
            st.session_state.pop(key, None)


    # --------------------------------------------------
    # DISPLAY UPLOADED IMAGE
    # --------------------------------------------------

    st.subheader("Uploaded Petri-dish Image")

    st.image(
        image,
        use_column_width=True
    )


    # --------------------------------------------------
    # AUTOMATIC YOLO DETECTION
    # --------------------------------------------------

    st.divider()

    st.subheader("Automatic Colony Detection")

    if st.button(
        "Run YOLO Automatic Detection",
        type="primary",
        use_container_width=True
    ):

        try:

            st.session_state["yolo_debug"] = "Button clicked successfully."

            with st.spinner("Loading YOLO model..."):

                model = load_model()

            st.session_state["yolo_debug"] += "\nYOLO model loaded successfully."

            with st.spinner("YOLO is detecting bacterial colonies..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    device="cpu",
                    verbose=False
                )

            result = results[0]

            detection_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            # Draw bounding boxes
            annotated_image = result.plot()

            # Convert BGR to RGB
            annotated_image = annotated_image[:, :, ::-1]

            # Count predicted classes
            class_counts = Counter()

            confidence_values = []

            yolo_objects = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())

                    class_name = model.names[class_id]

                    box_confidence = float(
                        box.conf[0].item()
                    )

                    # Bounding box coordinates
                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    x = max(0, int(round(x1)))
                    y = max(0, int(round(y1)))

                    width = max(
                        0,
                        int(round(x2 - x1))
                    )

                    height = max(
                        0,
                        int(round(y2 - y1))
                    )

                    class_counts[class_name] += 1

                    confidence_values.append({
                        "class": class_name,
                        "confidence": round(box_confidence, 3)
                    })

                    yolo_objects.append({

                        "class": class_name,

                        "height": height,

                        "id": len(yolo_objects) + 1,

                        "width": width,

                        "x": x,

                        "y": y,

                        "confidence": round(box_confidence, 3)
                    })


            # Store YOLO results
            st.session_state["yolo_image"] = annotated_image

            st.session_state["yolo_counts"] = dict(class_counts)

            st.session_state["yolo_total"] = detection_count

            st.session_state["yolo_confidences"] = confidence_values

            st.session_state["yolo_labels"] = yolo_objects

            st.session_state["yolo_classes"] = list(class_counts.keys())

            st.session_state["yolo_image_key"] = image_key

            st.session_state["yolo_debug"] += (
                f"\nPrediction completed."
                f"\nNumber of detections: {detection_count}"
                f"\nConfidence threshold: {confidence}"
                f"\nModel classes: {model.names}"
            )

        except Exception as e:

            st.session_state["yolo_debug"] = (
                f"YOLO detection failed: {str(e)}"
            )


    # --------------------------------------------------
    # YOLO DEBUG INFORMATION
    # --------------------------------------------------

    if "yolo_debug" in st.session_state:

        st.subheader("YOLO Debug Information")

        st.code(
            st.session_state["yolo_debug"]
        )


    # --------------------------------------------------
    # DISPLAY YOLO RESULTS
    # --------------------------------------------------

    if (
        st.session_state.get("yolo_image_key") == image_key
        and "yolo_image" in st.session_state
    ):

        st.divider()

        st.subheader("YOLO Detection Results")

        st.metric(
            "Total Automatically Detected Colonies",
            st.session_state["yolo_total"]
        )

        st.image(
            st.session_state["yolo_image"],
            caption="YOLOv8 Predicted Bounding Boxes",
            use_column_width=True
        )

        # Class distribution
        st.subheader("Predicted Class Distribution")

        counts = st.session_state["yolo_counts"]

        if counts:

            for class_name, count in counts.items():

                st.write(
                    f"**{class_name}:** {count}"
                )

        else:

            st.warning(
                "No colonies detected. "
                "Try lowering the confidence threshold."
            )

        # Confidence details
        st.subheader("Detection Confidence")

        confidence_values = st.session_state["yolo_confidences"]

        if confidence_values:

            st.dataframe(
                confidence_values,
                use_container_width=True
            )

        else:

            st.info(
                "No detection confidence values are available."
            )


        # --------------------------------------------------
        # YOLO PREDICTIONS JSON EXPORT
        # --------------------------------------------------

        st.subheader("Automatic YOLO Annotation JSON")

        yolo_output = {

            "background": "lower-resolution",

            "classes": st.session_state["yolo_classes"],

            "colonies_number": st.session_state["yolo_total"],

            "labels": st.session_state["yolo_labels"],

            "sample_id": Path(uploaded.name).stem,
        }

        yolo_json_text = json.dumps(
            yolo_output,
            indent=4
        )

        st.code(
            yolo_json_text,
            language="json"
        )

        st.download_button(
            label="Download YOLO Predictions JSON",
            data=yolo_json_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_yolo.json",
            mime="application/json",
            use_container_width=True
        )


    # --------------------------------------------------
    # MANUAL ANNOTATION
    # --------------------------------------------------

    st.divider()

    st.subheader("Manual Colony Annotation")

    st.caption(
        "Select the rectangle tool and draw one box around each colony. "
        "Use the canvas toolbar to undo or clear annotations."
    )

    max_width = 1000

    scale = min(
        1.0,
        max_width / image.width
    )

    display_size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale))
    )

    display_image = image.resize(display_size)

    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=display_image,
        update_streamlit=True,
        height=display_size[1],
        width=display_size[0],
        drawing_mode="rect",
        key=f"canvas_{image_key}",
        display_toolbar=True,
    )


    # --------------------------------------------------
    # PROCESS MANUAL ANNOTATIONS
    # --------------------------------------------------

    objects = []

    if (
        canvas_result.json_data
        and canvas_result.json_data.get("objects")
    ):

        for obj in canvas_result.json_data["objects"]:

            left = float(obj.get("left", 0)) / scale

            top = float(obj.get("top", 0)) / scale

            width = (
                float(obj.get("width", 0))
                * float(obj.get("scaleX", 1))
                / scale
            )

            height = (
                float(obj.get("height", 0))
                * float(obj.get("scaleY", 1))
                / scale
            )

            x = max(
                0,
                min(int(round(left)), image.width)
            )

            y = max(
                0,
                min(int(round(top)), image.height)
            )

            w = max(
                0,
                min(int(round(width)), image.width - x)
            )

            h = max(
                0,
                min(int(round(height)), image.height - y)
            )

            if w > 0 and h > 0:

                objects.append({

                    "class": colony_class,

                    "height": h,

                    "id": len(objects) + 1,

                    "width": w,

                    "x": x,

                    "y": y,
                })


    # --------------------------------------------------
    # MANUAL ANNOTATION SUMMARY
    # --------------------------------------------------

    st.subheader("Manual Annotation Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Manual Colony Boxes",
        len(objects)
    )

    c2.metric(
        "Image Width",
        image.width
    )

    c3.metric(
        "Image Height",
        image.height
    )


    # --------------------------------------------------
    # MANUAL JSON EXPORT
    # --------------------------------------------------

    output = {

        "background": "lower-resolution",

        "classes": [colony_class],

        "colonies_number": len(objects),

        "labels": objects,

        "sample_id": Path(uploaded.name).stem,
    }

    json_text = json.dumps(
        output,
        indent=4
    )

    st.subheader("Manual Annotation JSON Preview")

    st.code(
        json_text,
        language="json"
    )

    st.download_button(
        label="Download Manual Annotations JSON",
        data=json_text.encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_manual.json",
        mime="application/json",
        use_container_width=True,
    )


    # --------------------------------------------------
    # INFORMATION
    # --------------------------------------------------

    st.info(
        "Automatic detection uses YOLOv8 predictions. "
        "Manual annotation is also available. "
        "YOLO counts are model estimates and are not laboratory-validated CFU counts."
    )


else:

    st.info(
        "Upload an image to start."
    )
