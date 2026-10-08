import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import cv2


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
# COLOURS
# ============================================================

BG = "#FFF9F2"
CARD = "#FFFFFF"
TEXT = "#4A3F35"
SUBTEXT = "#81756B"

PINK = "#E89AAF"
PINK_HOVER = "#D97F98"
BLUE = "#A9D8E8"
GREEN = "#B9D9B0"
YELLOW = "#F8D98B"


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background: {BG};
    }}

    .main-title {{
        text-align: center;
        color: {TEXT};
        font-size: 42px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }}

    .subtitle {{
        text-align: center;
        color: {SUBTEXT};
        font-size: 17px;
        margin-bottom: 30px;
    }}

    .section-title {{
        color: {TEXT};
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 8px;
    }}

    .status {{
        text-align: center;
        color: {SUBTEXT};
        font-size: 15px;
        padding: 10px;
    }}

    div[data-testid="stFileUploader"] {{
        background: white;
        border-radius: 15px;
        padding: 8px;
    }}

    .result-box {{
        background: white;
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 3px 15px rgba(74,63,53,0.08);
    }}

    .info-box {{
        background: white;
        border-radius: 15px;
        padding: 15px;
        color: {TEXT};
        box-shadow: 0 2px 10px rgba(74,63,53,0.06);
    }}

    @media (max-width: 768px) {{
        .main-title {{
            font-size: 30px;
        }}

        .subtitle {{
            font-size: 14px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "reference_image" not in st.session_state:
    st.session_state.reference_image = None

if "palette_image" not in st.session_state:
    st.session_state.palette_image = None

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "available_palette" not in st.session_state:
    st.session_state.available_palette = []


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(uploaded_file):
    """Convert Streamlit uploaded/camera image into RGB PIL image."""

    if uploaded_file is None:
        return None

    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        return None


def resize_for_processing(image, max_dimension=900):
    """Resize large images while keeping aspect ratio."""

    image = image.copy()

    width, height = image.size

    largest = max(width, height)

    if largest <= max_dimension:
        return image

    scale = max_dimension / largest

    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale))
    )

    return image.resize(
        new_size,
        Image.Resampling.LANCZOS
    )


# ============================================================
# COLOUR DISTANCE
# ============================================================

def colour_distance_lab(colour1, colour2):
    """
    Calculate perceptual distance between two RGB colours
    using LAB colour space.
    """

    a = np.uint8([[colour1]])
    b = np.uint8([[colour2]])

    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )

    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )

    return float(np.linalg.norm(lab_a - lab_b))


# ============================================================
# MERGE SIMILAR COLOURS
# ============================================================

def merge_similar_colors(colors, counts=None, threshold=14):
    """
    Merge colours that are perceptually very similar.
    """

    if not colors:
        return []

    if counts is None:
        counts = [1] * len(colors)

    order = np.argsort(counts)[::-1]

    merged = []

    for idx in order:

        colour = tuple(
            int(x) for x in colors[idx]
        )

        too_similar = False

        for existing in merged:

            distance = colour_distance_lab(
                colour,
                existing
            )

            if distance < threshold:
                too_similar = True
                break

        if not too_similar:
            merged.append(colour)

    return merged


# ============================================================
# PALETTE DETECTION
# ============================================================

def extract_palette_colors(image, max_colors=24):
    """
    Detect available paint colours.

    Unlike the old version, this intentionally keeps:
    - colourful paints
    - white
    - black
    - grey
    - muted colours
    """

    image = resize_for_processing(
        image,
        max_dimension=300
    )

    img = np.array(image.convert("RGB"))

    # Small image for faster processing
    small = cv2.resize(
        img,
        (180, 180),
        interpolation=cv2.INTER_AREA
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------------------------
    # Keep BOTH colourful and neutral colours
    # --------------------------------------------------------

    colourful = (
        (saturation >= 30) &
        (value >= 25) &
        (value <= 255)
    )

    dark_neutral = (
        (saturation < 45) &
        (value <= 75)
    )

    light_neutral = (
        (saturation < 45) &
        (value >= 190)
    )

    neutral = (
        dark_neutral |
        light_neutral
    )

    mask = colourful | neutral

    pixels = small[mask]

    # If the mask is too restrictive, use all pixels
    if len(pixels) < 200:
        pixels = small.reshape(-1, 3)

    # Limit number of pixels
    if len(pixels) > 20000:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(pixels),
            20000,
            replace=False
        )

        pixels = pixels[indices]

    pixels = np.float32(pixels)

    if len(pixels) < 3:
        return []

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    K = min(
        max_colors,
        max(4, len(pixels) // 700)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.5
    )

    try:

        _, labels, centers = cv2.kmeans(
            pixels,
            K,
            None,
            criteria,
            12,
            cv2.KMEANS_PP_CENTERS
        )

    except Exception:
        return []

    centers = np.uint8(
        np.clip(
            centers,
            0,
            255
        )
    )

    counts = np.bincount(
        labels.flatten(),
        minlength=K
    )

    order = np.argsort(
        counts
    )[::-1]

    ordered_colors = []
    ordered_counts = []

    for index in order:

        colour = tuple(
            int(x)
            for x in centers[index]
        )

        ordered_colors.append(colour)
        ordered_counts.append(
            int(counts[index])
        )

    # Merge nearly identical colours
    detected = merge_similar_colors(
        ordered_colors,
        ordered_counts,
        threshold=13
    )

    # Limit number of paints
    detected = detected[:max_colors]

    return detected


# ============================================================
# REFERENCE COLOUR QUANTIZATION
# ============================================================

def create_reference_clusters(image, max_colors=18):
    """
    Break the reference painting into meaningful colour regions.

    This is more stable than repeatedly applying a distance mask
    to every colour centre.
    """

    image = resize_for_processing(
        image,
        max_dimension=850
    )

    img = np.array(
        image.convert("RGB")
    )

    height, width = img.shape[:2]

    # Work on a smaller image for K-means
    scale_width = min(width, 320)
    scale_height = max(
        1,
        int(height * (scale_width / width))
    )

    small = cv2.resize(
        img,
        (scale_width, scale_height),
        interpolation=cv2.INTER_AREA
    )

    pixels = small.reshape(
        (-1, 3)
    ).astype(
        np.float32
    )

    # Keep K reasonable
    K = min(
        max_colors,
        max(
            8,
            len(pixels) // 2500
        )
    )

    K = max(
        4,
        min(K, 18)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        70,
        0.4
    )

    _, _, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(
        np.clip(
            centers,
            0,
            255
        )
    )

    return centers


# ============================================================
# CLOSEST PAINT
# ============================================================

def closest_paint(reference_colour, palette):
    """
    Find the closest available paint in LAB space.
    """

    if not palette:
        return tuple(
            int(x)
            for x in reference_colour
        )

    ref = np.uint8(
        [[
            reference_colour
        ]]
    )

    ref_lab = cv2.cvtColor(
        ref,
        cv2.COLOR_RGB2LAB
    )[0, 0].astype(
        np.float32
    )

    best_colour = palette[0]
    best_distance = float("inf")

    for paint in palette:

        paint_array = np.uint8(
            [[paint]]
        )

        paint_lab = cv2.cvtColor(
            paint_array,
            cv2.COLOR_RGB2LAB
        )[0, 0].astype(
            np.float32
        )

        distance = np.linalg.norm(
            ref_lab - paint_lab
        )

        if distance < best_distance:

            best_distance = distance
            best_colour = paint

    return best_colour


# ============================================================
# GENERATE PAINTING
# ============================================================

def create_result(reference_image, palette):
    """
    Convert the reference into a painting using only
    colours available in the palette.
    """

    if reference_image is None:
        return None

    if not palette:
        return None

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    working_image = resize_for_processing(
        reference_image,
        max_dimension=900
    )

    img = np.array(
        working_image.convert("RGB")
    )

    height, width = img.shape[:2]

    # --------------------------------------------------------
    # Reference colour clusters
    # --------------------------------------------------------

    centers = create_reference_clusters(
        working_image,
        max_colors=18
    )

    # --------------------------------------------------------
    # Convert reference + centres to LAB
    # --------------------------------------------------------

    lab_image = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    ).astype(
        np.float32
    )

    center_rgb = np.uint8(
        centers
    )

    center_lab = cv2.cvtColor(
        center_rgb.reshape(
            1,
            -1,
            3
        ),
        cv2.COLOR_RGB2LAB
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Map every reference cluster to closest paint
    # --------------------------------------------------------

    palette_array = np.uint8(
        palette
    )

    palette_lab = cv2.cvtColor(
        palette_array.reshape(
            1,
            -1,
            3
        ),
        cv2.COLOR_RGB2LAB
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    cluster_to_paint = []

    for ref_lab in center_lab:

        distances = np.linalg.norm(
            palette_lab - ref_lab,
            axis=1
        )

        best_index = int(
            np.argmin(distances)
        )

        cluster_to_paint.append(
            palette[best_index]
        )

    # --------------------------------------------------------
    # Assign each pixel to closest reference cluster
    # --------------------------------------------------------

    flat_lab = lab_image.reshape(
        -1,
        3
    )

    # Process in chunks so large images don't use excessive RAM
    labels = np.empty(
        len(flat_lab),
        dtype=np.int32
    )

    chunk_size = 50000

    for start in range(
        0,
        len(flat_lab),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(flat_lab)
        )

        chunk = flat_lab[start:end]

        distances = np.linalg.norm(
            chunk[:, None, :] -
            center_lab[None, :, :],
            axis=2
        )

        labels[start:end] = np.argmin(
            distances,
            axis=1
        )

    # --------------------------------------------------------
    # Create result
    # --------------------------------------------------------

    result_flat = np.zeros(
        (
            len(flat_lab),
            3
        ),
        dtype=np.uint8
    )

    for i, paint in enumerate(
        cluster_to_paint
    ):

        result_flat[
            labels == i
        ] = np.array(
            paint,
            dtype=np.uint8
        )

    result = result_flat.reshape(
        height,
        width,
        3
    )

    result_image = Image.fromarray(
        result,
        "RGB"
    )

    # --------------------------------------------------------
    # Slight smoothing
    # --------------------------------------------------------

    result_image = result_image.filter(
        ImageFilter.GaussianBlur(
            radius=0.45
        )
    )

    return result_image


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎨 Smart Painting Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Turn your available colours into beautiful paintings ✨'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(
    2,
    gap="large"
)


# ============================================================
# REFERENCE
# ============================================================

with col1:

    st.markdown(
        '<div class="section-title">🖼️ Reference Painting</div>',
        unsafe_allow_html=True
    )

    reference_method = st.radio(
        "Reference input",
        [
            "📁 Gallery",
            "📷 Camera"
        ],
        horizontal=True,
        key="reference_method"
    )

    reference_file = None

    if reference_method == "📁 Gallery":

        reference_file = st.file_uploader(
            "Upload your reference painting",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp"
            ],
            key="reference_upload"
        )

    else:

        reference_file = st.camera_input(
            "Take a photo of your reference painting",
            key="reference_camera"
        )

    if reference_file is not None:

        new_reference = load_image(
            reference_file
        )

        if new_reference is not None:

            st.session_state.reference_image = (
                new_reference
            )

    if st.session_state.reference_image is not None:

        st.image(
            st.session_state.reference_image,
            use_container_width=True
        )


# ============================================================
# PALETTE
# ============================================================

with col2:

    st.markdown(
        '<div class="section-title">🎨 Your Paint Palette</div>',
        unsafe_allow_html=True
    )

    palette_method = st.radio(
        "Palette input",
        [
            "📁 Gallery",
            "📷 Camera"
        ],
        horizontal=True,
        key="palette_method"
    )

    palette_file = None

    if palette_method == "📁 Gallery":

        palette_file = st.file_uploader(
            "Upload your paint palette",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp"
            ],
            key="palette_upload"
        )

    else:

        palette_file = st.camera_input(
            "Take a photo of your paint palette",
            key="palette_camera"
        )

    if palette_file is not None:

        new_palette = load_image(
            palette_file
        )

        if new_palette is not None:

            # Avoid unnecessary reprocessing
            if (
                st.session_state.palette_image is None
                or new_palette.tobytes()
                != st.session_state.palette_image.tobytes()
            ):

                st.session_state.palette_image = (
                    new_palette
                )

                st.session_state.available_palette = (
                    extract_palette_colors(
                        new_palette
                    )
                )

    if st.session_state.palette_image is not None:

        st.image(
            st.session_state.palette_image,
            use_container_width=True
        )


# ============================================================
# DETECTED PALETTE
# ============================================================

if st.session_state.available_palette:

    st.markdown(
        '<div class="section-title">🎨 Detected Paint Colours</div>',
        unsafe_allow_html=True
    )

    palette_cols = st.columns(
        len(st.session_state.available_palette)
    )

    for col, colour in zip(
        palette_cols,
        st.session_state.available_palette
    ):

        hex_colour = "#{:02X}{:02X}{:02X}".format(
            *colour
        )

        col.markdown(
            f"""
            <div style="
                width:100%;
                height:45px;
                background:{hex_colour};
                border-radius:12px;
                border:1px solid #DDD;
                margin-bottom:5px;
            "></div>
            <div style="
                text-align:center;
                color:{TEXT};
                font-size:11px;
            ">
                {hex_colour}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="status">'
        f'{len(st.session_state.available_palette)} '
        f'paint colours detected ✓'
        f'</div>',
        unsafe_allow_html=True
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

st.markdown("")

generate_col1, generate_col2, generate_col3 = st.columns(
    [1, 2, 1]
)

with generate_col2:

    generate = st.button(
        "✨ Generate Painting",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# GENERATE
# ============================================================

if generate:

    if st.session_state.reference_image is None:

        st.warning(
            "Please upload or photograph a reference painting first."
        )

    elif st.session_state.palette_image is None:

        st.warning(
            "Please upload or photograph your paint palette first."
        )

    elif not st.session_state.available_palette:

        st.warning(
            "No paint colours could be detected from the palette."
        )

    else:

        with st.spinner(
            "🎨 Creating your painting..."
        ):

            try:

                result = create_result(
                    st.session_state.reference_image,
                    st.session_state.available_palette
                )

                st.session_state.result_image = result

            except Exception as e:

                st.error(
                    f"Could not generate the painting: {e}"
                )


# ============================================================
# RESULT
# ============================================================

if st.session_state.result_image is not None:

    st.markdown(
        '<div class="section-title">✨ Generated Painting</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.image(
        st.session_state.result_image,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    from io import BytesIO

    buffer = BytesIO()

    st.session_state.result_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="💾 Download Painting",
        data=buffer.getvalue(),
        file_name="smart_painting_result.png",
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# CLEAR
# ============================================================

st.markdown("")

if st.button(
    "🧹 Clear Project",
    use_container_width=True
):

    st.session_state.reference_image = None
    st.session_state.palette_image = None
    st.session_state.result_image = None
    st.session_state.available_palette = []

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="status">Create • Experiment • Paint 🎨</div>',
    unsafe_allow_html=True
)



        
