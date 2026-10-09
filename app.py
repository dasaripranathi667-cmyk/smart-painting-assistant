import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Painting Assistant",
    page_icon="🎨",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #fff7f2;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #5b4054;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #8d7085;
    font-size: 18px;
    margin-bottom: 35px;
}

.card {
    background-color: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.palette-box {
    display: inline-block;
    width: 48px;
    height: 48px;
    border-radius: 12px;
    margin: 5px;
    border: 2px solid white;
    box-shadow: 0 2px 7px rgba(0,0,0,0.12);
}

.project-card {
    background: white;
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "reference_image" not in st.session_state:
    st.session_state.reference_image = None

if "palette_image" not in st.session_state:
    st.session_state.palette_image = None

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "detected_palette" not in st.session_state:
    st.session_state.detected_palette = []

if "saved_projects" not in st.session_state:
    st.session_state.saved_projects = []

if "selected_project" not in st.session_state:
    st.session_state.selected_project = None


# ============================================================
# NAVIGATION
# ============================================================

def go_home():
    st.session_state.page = "home"


def go_new_project():
    st.session_state.page = "new"


def go_saved_projects():
    st.session_state.page = "saved"


def clear_current_project():

    st.session_state.reference_image = None
    st.session_state.palette_image = None
    st.session_state.result_image = None
    st.session_state.detected_palette = []


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_bytes(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ============================================================
# COLOR DISTANCE
# ============================================================

def color_distance_lab(color1, color2):

    a = np.uint8([[color1]])
    b = np.uint8([[color2]])

    lab1 = cv2.cvtColor(
        a,
        cv2.COLOR_RGB2LAB
    )[0, 0].astype(float)

    lab2 = cv2.cvtColor(
        b,
        cv2.COLOR_RGB2LAB
    )[0, 0].astype(float)

    return np.linalg.norm(
        lab1 - lab2
    )


# ============================================================
# MERGE SIMILAR COLORS
# ============================================================

def merge_similar_colors(
    colors,
    threshold=12
):

    merged = []

    for color in colors:

        if not merged:
            merged.append(color)
            continue

        distances = [
            color_distance_lab(
                color,
                existing
            )
            for existing in merged
        ]

        if min(distances) > threshold:
            merged.append(color)

    return merged


# ============================================================
# EXTRACT PALETTE COLORS
# ============================================================
def extract_palette_colors(image, max_colors=20):

    image = image.convert("RGB")

    img = np.array(image)

    # Small image for faster processing
    img = cv2.resize(
        img,
        (150, 150),
        interpolation=cv2.INTER_AREA
    )

    pixels = img.reshape(
        (-1, 3)
    ).astype(np.float32)

    # Sample pixels for speed
    if len(pixels) > 6000:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(pixels),
            6000,
            replace=False
        )

        pixels = pixels[indices]

    # Start with candidate clusters.
    # This is NOT the final number of colors.
    k = min(
        max_colors,
        len(pixels)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    counts = np.bincount(
        labels.flatten(),
        minlength=k
    )

    # Sort by how much of the image each color occupies
    order = np.argsort(
        counts
    )[::-1]

    candidate_colors = []

    for index in order:

        # Ignore extremely tiny regions
        # caused by reflections/noise.
        if counts[index] < len(pixels) * 0.005:
            continue

        candidate_colors.append(
            centers[index].tolist()
        )

    # --------------------------------------------------------
    # Merge visually similar colors
    # --------------------------------------------------------

    final_colors = []

    for color in candidate_colors:

        if not final_colors:

            final_colors.append(color)
            continue

        distances = [
            color_distance_lab(
                color,
                existing
            )
            for existing in final_colors
        ]

        # 18-20 is a good starting point.
        # Higher = more aggressive merging.
        if min(distances) > 18:

            final_colors.append(color)

    return final_colors


# ============================================================
# CREATE REFERENCE CLUSTERS
# ============================================================

def create_reference_clusters(
    image,
    max_colors=18
):

    img = np.array(
        image.convert("RGB")
    )

    height, width = img.shape[:2]

    scale = min(
        1.0,
        700 / max(height, width)
    )

    if scale < 1:

        img = cv2.resize(
            img,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    pixels = img.reshape(
        (-1, 3)
    ).astype(np.float32)

    pixels = np.round(
        pixels / 6
    ) * 6

    pixels = np.clip(
        pixels,
        0,
        255
    )

    k = min(
        max_colors,
        len(pixels)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        35,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    return centers


# ============================================================
# FIND CLOSEST PAINT COLOR
# ============================================================

def closest_paint(
    reference_color,
    palette
):

    if not palette:
        return reference_color

    distances = []

    for paint in palette:

        distances.append(
            color_distance_lab(
                reference_color,
                paint
            )
        )

    index = int(
        np.argmin(distances)
    )

    return palette[index]
    
# ============================================================
# SHADOW CORRECTION
# ============================================================

def correct_shadow_pixels(image):
    """
    Reduce false grey detections in dark shadow regions
    when nearby pixels contain genuine paint colours.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # Detect pixels that are both dark and nearly grey
    shadow_candidates = (
        (brightness < 115) &
        (saturation < 65)
    ).astype(np.uint8) * 255

    # Find colourful pixels that can help identify
    # the likely colour beneath a shadow
    colourful_pixels = (
        (saturation > 70) &
        (brightness > 80)
    ).astype(np.uint8) * 255

    # Look for colourful pixels in the nearby region
    kernel = np.ones((15, 15), np.uint8)

    nearby_colour = cv2.dilate(
        colourful_pixels,
        kernel,
        iterations=1
    )

    # Correct only dark, neutral pixels near colour
    shadow_mask = cv2.bitwise_and(
        shadow_candidates,
        nearby_colour
    )

    # Fill shadow pixels using surrounding image colours
    corrected = cv2.inpaint(
        image,
        shadow_mask,
        5,
        cv2.INPAINT_TELEA
    )

    return corrected


# ============================================================
# CREATE PAINTING RESULT
# ============================================================

def create_result(
    reference_image,
    palette
):

    original = reference_image.convert(
        "RGB"
    )

    original_array = np.array(
        original
    )

    height, width = original_array.shape[:2]

    max_dimension = 700

    scale = min(
        1.0,
        max_dimension / max(height, width)
    )

    if scale < 1:

        working = cv2.resize(
            original_array,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        working = original_array.copy()
        
    # --------------------------------------------------------
    # Correct likely shadows before detecting colours
    # --------------------------------------------------------

    working = correct_shadow_pixels(working)

   
    # --------------------------------------------------------
    # Reference color clusters
    # --------------------------------------------------------

    clusters = create_reference_clusters(
        Image.fromarray(working),
        max_colors=18
    )

    # --------------------------------------------------------
    # Convert image and clusters to LAB
    # --------------------------------------------------------

    lab_image = cv2.cvtColor(
        working,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    lab_clusters = cv2.cvtColor(
        clusters.reshape((-1, 1, 3)),
        cv2.COLOR_RGB2LAB
    ).reshape(
        (-1, 3)
    ).astype(np.float32)

    pixels = lab_image.reshape(
        (-1, 3)
    )

    # --------------------------------------------------------
    # Find nearest reference cluster for every pixel
    # --------------------------------------------------------

    distances = np.zeros(
        (
            len(pixels),
            len(lab_clusters)
        ),
        dtype=np.float32
    )

    for i, cluster in enumerate(
        lab_clusters
    ):

        distances[:, i] = np.linalg.norm(
            pixels - cluster,
            axis=1
        )

    labels = np.argmin(
        distances,
        axis=1
    )

    # --------------------------------------------------------
    # Replace each cluster with nearest paint color
    # --------------------------------------------------------

    output = np.zeros_like(
        working
    )

    output_pixels = output.reshape(
        (-1, 3)
    )

    for cluster_index in range(
        len(clusters)
    ):

        mask = (
            labels == cluster_index
        )

        if not np.any(mask):
            continue

        reference_color = (
            clusters[cluster_index]
        )

        paint_color = closest_paint(
            reference_color,
            palette
        )

        output_pixels[mask] = (
            paint_color
        )

    # --------------------------------------------------------
    # Slight smoothing
    # --------------------------------------------------------

    output = cv2.GaussianBlur(
        output,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Restore original size
    # --------------------------------------------------------

    if scale < 1:

        output = cv2.resize(
            output,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    return Image.fromarray(
        output
    )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        '<div class="main-title">'
        '🎨 Smart Painting Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Turn your reference image into a '
        'paint-friendly version.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    # --------------------------------------------------------
    # New Project
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card">
                <h2>✨ New Project</h2>
                <p>
                Start a new painting using your
                reference image and paint palette.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🎨 NEW PROJECT",
            use_container_width=True
        ):

            go_new_project()
            st.rerun()

    # --------------------------------------------------------
    # Saved Projects
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="card">
                <h2>📁 Saved Projects</h2>
                <p>
                View and reopen your previously
                saved paintings.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "📂 SAVED PROJECTS",
            use_container_width=True
        ):

            go_saved_projects()
            st.rerun()

    st.write("")
    st.write("")

    project_count = len(
        st.session_state.saved_projects
    )

    st.info(
        f"📁 {project_count} project(s) "
        "saved in this session."
    )


# ============================================================
# NEW PROJECT PAGE
# ============================================================

def new_project_page():

    col_back, col_title = st.columns(
        [1, 5]
    )

    with col_back:

        if st.button("← Home"):

            go_home()
            st.rerun()

    with col_title:

        st.markdown(
            "<h1>🎨 New Project</h1>",
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================
    # REFERENCE IMAGE
    # ========================================================

    st.subheader(
        "1. Reference Image"
    )

    reference_file = st.file_uploader(
        "Upload your reference image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="reference_uploader"
    )

    if reference_file is not None:

        st.session_state.reference_image = (
            Image.open(
                reference_file
            ).convert("RGB")
        )

    if st.session_state.reference_image:

        st.image(
            st.session_state.reference_image,
            caption="Reference Image",
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # PALETTE IMAGE
    # ========================================================

    st.subheader(
        "2. Paint Palette"
    )

    palette_file = st.file_uploader(
        "Upload a photo of your paint palette",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="palette_uploader"
    )

    if palette_file is not None:

        st.session_state.palette_image = (
            Image.open(
                palette_file
            ).convert("RGB")
        )

        # Detect colors immediately
        st.session_state.detected_palette = (
            extract_palette_colors(
                st.session_state.palette_image,
                max_colors=20
            )
        )

    if st.session_state.palette_image:

        st.image(
            st.session_state.palette_image,
            caption="Paint Palette",
            use_container_width=True
        )

    # ========================================================
    # DETECTED COLORS
    # ========================================================

    if st.session_state.detected_palette:

        st.subheader(
            "Detected Paint Colors"
        )

        html = ""

        for color in (
            st.session_state.detected_palette
        ):

            r, g, b = color

            html += (
                f'<span class="palette-box" '
                f'style="background:rgb({r},{g},{b})" '
                f'title="RGB {r},{g},{b}"></span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

        st.caption(
            f"{len(st.session_state.detected_palette)} "
            "colors detected"
        )

    st.divider()

    # ========================================================
    # GENERATE
    # ========================================================

    if st.button(
        "✨ GENERATE PAINTING",
        type="primary",
        use_container_width=True
    ):

        if (
            st.session_state.reference_image
            is None
        ):

            st.warning(
                "Please upload a reference image."
            )

        elif (
            st.session_state.palette_image
            is None
        ):

            st.warning(
                "Please upload your paint palette."
            )

        elif not (
            st.session_state.detected_palette
        ):

            st.warning(
                "No paint colors were detected."
            )

        else:

            with st.spinner(
                "Creating your painting..."
            ):

                st.session_state.result_image = (
                    create_result(
                        st.session_state.reference_image,
                        st.session_state.detected_palette
                    )
                )

            st.success(
                "Painting generated successfully!"
            )

    # ========================================================
    # RESULT
    # ========================================================

    if st.session_state.result_image:

        st.divider()

        st.subheader(
            "3. Generated Painting"
        )

        st.image(
            st.session_state.result_image,
            caption="Generated Painting",
            use_container_width=True
        )

        st.download_button(
            "⬇️ DOWNLOAD PAINTING",
            data=image_to_bytes(
                st.session_state.result_image
            ),
            file_name="smart_painting.png",
            mime="image/png",
            use_container_width=True
        )

        st.divider()

        # ====================================================
        # SAVE PROJECT
        # ====================================================

        st.subheader(
            "💾 Save Project"
        )

        project_name = st.text_input(
            "Project name",
            placeholder="Example: Sunset Landscape",
            key="project_name"
        )

        if st.button(
            "💾 SAVE PROJECT",
            use_container_width=True
        ):

            if not project_name.strip():

                st.warning(
                    "Please enter a project name."
                )

            else:

                new_project = {

                    "name":
                        project_name.strip(),

                    "reference":
                        st.session_state.reference_image.copy(),

                    "palette":
                        st.session_state.palette_image.copy(),

                    "result":
                        st.session_state.result_image.copy(),

                    "colors":
                        list(
                            st.session_state.detected_palette
                        ),

                    "date":
                        datetime.now().strftime(
                            "%d %b %Y, %I:%M %p"
                        )
                }

                st.session_state.saved_projects.append(
                    new_project
                )

                st.success(
                    f"✅ '{project_name.strip()}' "
                    "saved successfully!"
                )

    st.divider()

    # ========================================================
    # CLEAR
    # ========================================================

    if st.button(
        "🗑️ CLEAR PROJECT",
        use_container_width=True
    ):

        clear_current_project()

        # Clear uploader widgets
        st.session_state.pop(
            "reference_uploader",
            None
        )

        st.session_state.pop(
            "palette_uploader",
            None
        )

        st.session_state.pop(
            "project_name",
            None
        )

        st.rerun()


# ============================================================
# SAVED PROJECTS PAGE
# ============================================================

def saved_projects_page():

    col_back, col_title = st.columns(
        [1, 5]
    )

    with col_back:

        if st.button("← Home"):

            go_home()
            st.rerun()

    with col_title:

        st.markdown(
            "<h1>📁 Saved Projects</h1>",
            unsafe_allow_html=True
        )

    st.divider()

    projects = (
        st.session_state.saved_projects
    )

    # --------------------------------------------------------
    # No projects
    # --------------------------------------------------------

    if not projects:

        st.info(
            "You don't have any saved projects yet."
        )

        st.write("")

        if st.button(
            "🎨 CREATE NEW PROJECT",
            use_container_width=True
        ):

            go_new_project()
            st.rerun()

        return

    st.write(
        f"**{len(projects)} saved project(s)**"
    )

    st.write("")

    # --------------------------------------------------------
    # Project grid
    # --------------------------------------------------------

    columns = st.columns(3)

    for index, project in enumerate(
        projects
    ):

        with columns[
            index % 3
        ]:

            st.markdown(
                '<div class="project-card">',
                unsafe_allow_html=True
            )

            st.image(
                project["result"],
                use_container_width=True
            )

            st.markdown(
                f"### {project['name']}"
            )

            st.caption(
                f"🕒 {project['date']}"
            )

            col_open, col_delete = st.columns(
                2
            )

            # ----------------------------------------------
            # OPEN
            # ----------------------------------------------

            with col_open:

                if st.button(
                    "👁️ Open",
                    key=f"open_{index}",
                    use_container_width=True
                ):

                    st.session_state.selected_project = (
                        index
                    )

                    st.session_state.page = (
                        "view_project"
                    )

                    st.rerun()

            # ----------------------------------------------
            # DELETE
            # ----------------------------------------------

            with col_delete:

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{index}",
                    use_container_width=True
                ):

                    st.session_state.saved_projects.pop(
                        index
                    )

                    st.success(
                        "Project deleted."
                    )

                    st.rerun()

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# VIEW SAVED PROJECT
# ============================================================

def view_project_page():

    index = (
        st.session_state.selected_project
    )

    projects = (
        st.session_state.saved_projects
    )

    # Safety check
    if (
        index is None
        or index >= len(projects)
    ):

        go_saved_projects()
        st.rerun()

    project = projects[index]

    # --------------------------------------------------------
    # Back
    # --------------------------------------------------------

    if st.button(
        "← Back to Saved Projects"
    ):

        go_saved_projects()
        st.rerun()

    st.title(
        f"🎨 {project['name']}"
    )

    st.caption(
        f"Saved: {project['date']}"
    )

    st.divider()

    # --------------------------------------------------------
    # Reference + palette
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Reference Image"
        )

        st.image(
            project["reference"],
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Paint Palette"
        )

        st.image(
            project["palette"],
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # Generated painting
    # --------------------------------------------------------

    st.subheader(
        "Generated Painting"
    )

    st.image(
        project["result"],
        use_container_width=True
    )

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    if project["colors"]:

        st.subheader(
            "Paint Colors Used"
        )

        html = ""

        for color in project["colors"]:

            r, g, b = color

            html += (
                f'<span class="palette-box" '
                f'style="background:rgb({r},{g},{b})" '
                f'title="RGB {r},{g},{b}"></span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    st.download_button(
        "⬇️ DOWNLOAD PAINTING",
        data=image_to_bytes(
            project["result"]
        ),
        file_name=(
            project["name"]
            .replace(" ", "_")
            + ".png"
        ),
        mime="image/png",
        use_container_width=True
    )

    st.write("")

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    if st.button(
        "🗑️ DELETE THIS PROJECT",
        use_container_width=True
    ):

        st.session_state.saved_projects.pop(
            index
        )

        st.session_state.selected_project = None

        go_saved_projects()

        st.rerun()


# ============================================================
# PAGE ROUTER
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "new":

    new_project_page()

elif st.session_state.page == "saved":

    saved_projects_page()

elif st.session_state.page == "view_project":

    view_project_page()



        
