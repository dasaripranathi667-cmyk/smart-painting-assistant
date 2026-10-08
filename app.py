import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import cv2
import io


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Painting Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "reference_image": None,
    "palette_image": None,
    "result_image": None,
    "available_palette": [],
    "reference_source": None,
    "palette_source": None
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# COLOURS
# ============================================================

BG = "#FFF9F2"
CARD = "#FFFFFF"
TEXT = "#4A3F35"
SUBTEXT = "#81756B"

PINK = "#F6B6C8"
PINK_DARK = "#D97F98"

BLUE = "#A9D8E8"
BLUE_DARK = "#8CC5D8"

GREEN = "#B9D9B0"
GREEN_DARK = "#A4C998"

YELLOW = "#F8D98B"
YELLOW_DARK = "#EBCB76"

CREAM = "#F5F0EA"


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
<style>

.stApp {{
    background-color: {BG};
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}


/* Main headings */

.app-title {{
    text-align: center;
    color: {TEXT};
    font-size: 42px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 5px;
}}

.app-subtitle {{
    text-align: center;
    color: {SUBTEXT};
    font-size: 17px;
    margin-bottom: 30px;
}}


/* Cards */

.card {{
    background: {CARD};
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 4px 18px rgba(74,63,53,0.08);
    border: 1px solid rgba(74,63,53,0.04);
}}


/* Section headings */

.section-heading {{
    color: {TEXT};
    font-size: 21px;
    font-weight: 650;
    margin-bottom: 12px;
}}


/* Home icon */

.paint-icon {{
    text-align: center;
    font-size: 70px;
    margin-top: 20px;
}}


/* Buttons */

.stButton > button {{
    border-radius: 14px;
    min-height: 48px;
    font-weight: 650;
    border: none;
}}


/* File upload */

[data-testid="stFileUploader"] {{
    background-color: {CREAM};
    border-radius: 15px;
    padding: 8px;
}}


/* Camera */

[data-testid="stCameraInput"] {{
    border-radius: 15px;
}}


/* Image */

[data-testid="stImage"] img {{
    border-radius: 18px;
}}


/* Status */

.status-box {{
    background: {CARD};
    border-radius: 15px;
    padding: 12px;
    text-align: center;
    color: {SUBTEXT};
    margin-top: 15px;
}}


/* Result */

.result-heading {{
    text-align: center;
    color: {TEXT};
    font-size: 27px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}}


/* Count */

.count-box {{
    background: #FDF4E5;
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    color: {TEXT};
    font-weight: 600;
}}


/* Mobile */

@media (max-width: 768px) {{

    .app-title {{
        font-size: 30px;
    }}

    .app-subtitle {{
        font-size: 14px;
    }}

    .paint-icon {{
        font-size: 55px;
    }}

}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# PALETTE DETECTION
# ============================================================

def fallback_palette_detection(img):

    small = cv2.resize(
        img,
        (100, 100)
    )

    pixels = small.reshape(
        (-1, 3)
    ).astype(np.float32)

    K = 8

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    counts = np.bincount(
        labels.flatten(),
        minlength=K
    )

    order = np.argsort(
        counts
    )[::-1]

    detected = []

    for index in order:

        colour = tuple(
            int(x)
            for x in centers[index]
        )

        too_similar = False

        for existing in detected:

            distance = np.linalg.norm(
                np.array(colour, dtype=float)
                -
                np.array(existing, dtype=float)
            )

            if distance < 35:
                too_similar = True
                break

        if not too_similar:

            detected.append(
                colour
            )

    return detected


def extract_palette_colors(image):

    img = np.array(
        image.convert("RGB")
    )

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colourful_mask = (
        (saturation > 45) &
        (value > 45) &
        (value < 250)
    )

    ys, xs = np.where(
        colourful_mask
    )

    if len(xs) < 100:

        return fallback_palette_detection(
            img
        )

    pixels = img[ys, xs]

    if len(pixels) > 15000:

        indices = np.random.choice(
            len(pixels),
            15000,
            replace=False
        )

        pixels = pixels[indices]

    pixels_float = np.float32(
        pixels
    )

    K = min(
        12,
        max(
            3,
            len(pixels_float) // 500
        )
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels_float,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(
        centers
    )

    counts = np.bincount(
        labels.flatten(),
        minlength=K
    )

    order = np.argsort(
        counts
    )[::-1]

    detected = []

    for index in order:

        colour = tuple(
            int(x)
            for x in centers[index]
        )

        too_similar = False

        for existing in detected:

            distance = np.linalg.norm(
                np.array(colour, dtype=float)
                -
                np.array(existing, dtype=float)
            )

            if distance < 35:

                too_similar = True
                break

        if not too_similar:

            detected.append(
                colour
            )

    return detected


# ============================================================
# REFERENCE COLOUR GROUPING
# ============================================================

def group_reference_colors(image):

    img = np.array(
        image.convert("RGB")
    )

    small = cv2.resize(
        img,
        (200, 200)
    )

    pixels = small.reshape(
        (-1, 3)
    ).astype(np.float32)

    K = min(
        18,
        max(
            6,
            len(pixels) // 1500
        )
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS
    )

    return np.uint8(
        centers
    )


# ============================================================
# CLOSEST PAINT
# ============================================================

def closest_paint(
    reference_colour,
    palette
):

    if not palette:

        return tuple(
            int(x)
            for x in reference_colour
        )

    reference = np.array(
        reference_colour,
        dtype=np.float32
    )

    best_colour = palette[0]

    best_distance = float(
        "inf"
    )

    for paint in palette:

        paint_array = np.array(
            paint,
            dtype=np.float32
        )

        distance = np.linalg.norm(
            reference -
            paint_array
        )

        if distance < best_distance:

            best_distance = distance
            best_colour = paint

    return best_colour


# ============================================================
# CREATE RESULT
# ============================================================

def create_result(
    reference_image,
    palette
):

    img = np.array(
        reference_image.convert("RGB")
    )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    centers = group_reference_colors(
        reference_image
    )

    result = img.copy()

    for center in centers:

        paint = closest_paint(
            center,
            palette
        )

        center_array = np.uint8(
            [[center]]
        )

        center_lab = cv2.cvtColor(
            center_array,
            cv2.COLOR_RGB2LAB
        )[0, 0]

        distance = np.linalg.norm(
            lab.astype(np.float32)
            -
            center_lab.astype(np.float32),
            axis=2
        )

        mask = distance < 35

        result[mask] = paint

    result_image = Image.fromarray(
        result
    )

    result_image = result_image.filter(
        ImageFilter.GaussianBlur(
            radius=0.6
        )
    )

    return result_image


# ============================================================
# RESET
# ============================================================

def reset_project():

    st.session_state.reference_image = None
    st.session_state.palette_image = None
    st.session_state.result_image = None
    st.session_state.available_palette = []

    st.session_state.reference_source = None
    st.session_state.palette_source = []


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        '<div class="paint-icon">🎨</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">'
        'Smart Painting Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Turn your available colours into beautiful paintings ✨'
        '</div>',
        unsafe_allow_html=True
    )

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.markdown(
            f"""
            <div class="card">

            <h2 style="text-align:center;color:{TEXT};">
            Paint Smarter
            </h2>

            <p style="text-align:center;color:{SUBTEXT};">
            Upload a reference painting and a photo of
            the paints you already have.
            </p>

            <p style="text-align:center;color:{SUBTEXT};">
            Smart Painting Assistant finds the closest
            available colours and creates a personalized
            painting preview.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "✨  NEW PROJECT",
            use_container_width=True
        ):

            st.session_state.page = "project"

            st.rerun()

        if st.button(
            "📂  SAVED PROJECTS",
            use_container_width=True
        ):

            st.session_state.page = "saved"

            st.rerun()

        st.write("")

        st.caption(
            "Create • Experiment • Paint 🎨"
        )


# ============================================================
# PROJECT PAGE
# ============================================================

def project_page():

    if st.button(
        "← Home"
    ):

        st.session_state.page = "home"

        st.rerun()

    st.markdown(
        '<div class="app-title">'
        'Create Your Painting'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Upload your reference and available colours'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # IMAGE CARDS
    # ========================================================

    reference_col, palette_col, result_col = st.columns(
        3
    )


    # ========================================================
    # REFERENCE CARD
    # ========================================================

    with reference_col:

        st.markdown(
            f"""
            <div class="card">

            <div class="section-heading">
            🖼️ Reference Painting
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        reference_upload = st.file_uploader(
            "🖼️ Choose from Gallery",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp"
            ],
            key="reference_upload"
        )

        reference_camera = st.camera_input(
            "📷 Take Photo",
            key="reference_camera"
        )

        selected_reference = (
            reference_camera
            if reference_camera is not None
            else reference_upload
        )

        if selected_reference is not None:

            image = Image.open(
                selected_reference
            ).convert("RGB")

            st.session_state.reference_image = image

        if st.session_state.reference_image is not None:

            st.image(
                st.session_state.reference_image,
                use_container_width=True
            )


    # ========================================================
    # PALETTE CARD
    # ========================================================

    with palette_col:

        st.markdown(
            f"""
            <div class="card">

            <div class="section-heading">
            🎨 Your Paint Palette
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        palette_upload = st.file_uploader(
            "🖼️ Choose from Gallery",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp"
            ],
            key="palette_upload"
        )

        palette_camera = st.camera_input(
            "📷 Take Photo",
            key="palette_camera"
        )

        selected_palette = (
            palette_camera
            if palette_camera is not None
            else palette_upload
        )

        if selected_palette is not None:

            image = Image.open(
                selected_palette
            ).convert("RGB")

            st.session_state.palette_image = image

            with st.spinner(
                "Detecting colours..."
            ):

                st.session_state.available_palette = (
                    extract_palette_colors(
                        image
                    )
                )

        if st.session_state.palette_image is not None:

            st.image(
                st.session_state.palette_image,
                use_container_width=True
            )

            count = len(
                st.session_state.available_palette
            )

            st.markdown(
                f"""
                <div class="count-box">
                🎨 {count} paint colours detected
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # RESULT CARD
    # ========================================================

    with result_col:

        st.markdown(
            f"""
            <div class="card">

            <div class="section-heading">
            ✨ Generated Painting
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.session_state.result_image is not None:

            st.image(
                st.session_state.result_image,
                use_container_width=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                background:{CREAM};
                border-radius:18px;
                min-height:250px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                color:{SUBTEXT};
                ">

                <div>
                ✨<br><br>
                Your generated painting<br>
                will appear here
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    st.write("")

    generate_col, clear_col = st.columns(
        [3, 1]
    )

    with generate_col:

        if st.button(
            "✨  GENERATE PAINTING",
            use_container_width=True
        ):

            if st.session_state.reference_image is None:

                st.warning(
                    "Please upload or take a reference painting photo."
                )

            elif st.session_state.palette_image is None:

                st.warning(
                    "Please upload or take a paint palette photo."
                )

            elif not st.session_state.available_palette:

                st.warning(
                    "No paint colours were detected."
                )

            else:

                with st.spinner(
                    "Creating your personalized painting..."
                ):

                    st.session_state.result_image = (
                        create_result(
                            st.session_state.reference_image,
                            st.session_state.available_palette
                        )
                    )

                st.success(
                    "Your painting has been generated! ✨"
                )

                st.rerun()

    with clear_col:

        if st.button(
            "🧹 CLEAR",
            use_container_width=True
        ):

            reset_project()

            st.rerun()


    # ========================================================
    # DOWNLOAD
    # ========================================================

    if st.session_state.result_image is not None:

        st.markdown(
            '<div class="result-heading">'
            '✨ Your Painting is Ready!'
            '</div>',
            unsafe_allow_html=True
        )

        image_bytes = io.BytesIO()

        st.session_state.result_image.save(
            image_bytes,
            format="PNG"
        )

        st.download_button(
            "💾  DOWNLOAD PAINTING",
            data=image_bytes.getvalue(),
            file_name="smart_painting_result.png",
            mime="image/png",
            use_container_width=True
        )


# ============================================================
# SAVED PROJECTS
# ============================================================

def saved_page():

    if st.button(
        "← Home"
    ):

        st.session_state.page = "home"

        st.rerun()

    st.markdown(
        '<div class="app-title">'
        '📂 Saved Projects'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Your future painting collection ✨'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Online project storage will be connected when "
        "we deploy the website."
    )


# ============================================================
# NAVIGATION
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "project":

    project_page()

elif st.session_state.page == "saved":

    saved_page()