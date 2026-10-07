# ui.py
# Streamlit User Interface Module
# Assignment 5 - Real-Time Object Detection & Logging Platform

from html import escape

import pandas as pd
import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    CONFIDENCE_STEP,
    DEFAULT_CONFIDENCE,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    OBJECT_CLASSES,
)


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
def show_header():
    """
    Display the main application header.
    """

    st.markdown(
        """
        <style>
        :root {
            --rt-accent: #22d3ee;
            --rt-accent-strong: #0891b2;
            --rt-ink: #e7eef8;
            --rt-muted: #9aabc1;
            --rt-line: rgba(148, 163, 184, .18);
            --rt-panel: rgba(19, 30, 48, .72);
        }
        .block-container {
            max-width: 1540px;
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
        }
        .rt-hero {
            position: relative;
            overflow: hidden;
            padding: clamp(1.4rem, 3vw, 2.35rem);
            border: 1px solid rgba(103, 232, 249, .2);
            border-radius: 24px;
            background:
                radial-gradient(ellipse at 88% 12%, rgba(34, 211, 238, .18), transparent 38%),
                linear-gradient(120deg, rgba(15, 34, 56, .98), rgba(17, 27, 45, .94) 58%, rgba(14, 45, 62, .9));
            box-shadow: 0 20px 55px rgba(2, 8, 23, .2);
            margin-bottom: 1.4rem;
        }
        .rt-hero-kicker {
            color: #67e8f9;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin-bottom: .75rem;
        }
        .rt-hero-title {
            color: #f4f8ff;
            font-size: clamp(1.7rem, 4vw, 2.75rem);
            font-weight: 780;
            letter-spacing: -.045em;
            line-height: 1.08;
            margin: 0;
        }
        .rt-hero-copy {
            max-width: 690px;
            color: #b5c4d8;
            font-size: 1rem;
            line-height: 1.65;
            margin-top: .85rem;
        }
        .rt-tech-row {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.2rem;
        }
        .rt-tech-pill {
            color: #d9faff;
            font-size: .74rem;
            font-weight: 700;
            border: 1px solid rgba(103, 232, 249, .2);
            border-radius: 999px;
            background: rgba(8, 145, 178, .14);
            padding: .35rem .72rem;
        }
        div[data-testid="stMetric"] {
            min-height: 112px;
            background: linear-gradient(145deg, rgba(35, 48, 68, .82), rgba(17, 28, 43, .82));
            border: 1px solid var(--rt-line);
            border-radius: 17px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 10px 25px rgba(2, 8, 23, .12);
            transition: transform .18s ease, border-color .18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(34, 211, 238, .38);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--rt-muted);
            font-size: .79rem;
            font-weight: 650;
            letter-spacing: .035em;
        }
        div[data-testid="stMetricValue"] {
            color: var(--rt-ink);
            font-size: clamp(1.45rem, 2vw, 2rem);
            font-weight: 750;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--rt-line);
        }
        section[data-testid="stSidebar"] h2 {
            letter-spacing: -.025em;
        }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
            color: var(--rt-muted);
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            box-shadow: 0 7px 18px rgba(8, 145, 178, .22);
        }
        div[data-testid="stAlert"] {
            border-radius: 13px;
            border-width: 1px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--rt-line);
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-testid="stImage"] img {
            border-radius: 13px;
        }
        h2, h3 {
            letter-spacing: -.025em;
        }
        @media (max-width: 720px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .rt-hero {
                border-radius: 18px;
            }
            div[data-testid="stMetric"] {
                min-height: 92px;
                padding: .85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<header class="rt-hero">'
        '<div class="rt-hero-kicker">Computer vision · Live analytics</div>'
        f'<h1 class="rt-hero-title">{escape(APP_ICON)} {escape(APP_TITLE)}</h1>'
        '<p class="rt-hero-copy">A live camera workspace for real-time object detection. '
        'Tune the confidence threshold, focus on a class, and review recent events.</p>'
        '<div class="rt-tech-row">'
        '<span class="rt-tech-pill">YOLO inference</span>'
        '<span class="rt-tech-pill">OpenCV video</span>'
        '<span class="rt-tech-pill">MySQL events</span>'
        '</div>'
        '</header>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
def show_sidebar():
    """
    Display detection configuration controls.

    Returns:
        confidence_threshold
        object_filter
    """

    st.sidebar.markdown("## ⚙️ Controls")
    st.sidebar.caption("Tune the detector before or during a session.")

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=MIN_CONFIDENCE,
        max_value=MAX_CONFIDENCE,
        value=DEFAULT_CONFIDENCE,
        step=CONFIDENCE_STEP,
        help="Detections below this confidence level are ignored.",
    )
    st.sidebar.caption(f"Minimum confidence: **{confidence_threshold:.0%}**")

    # Object filter
    object_filter = st.sidebar.selectbox(
        "Object filter",
        OBJECT_CLASSES,
        help="Choose one class or process every class detected by YOLO.",
    )

    st.sidebar.divider()

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = True

    start_col, stop_col = st.sidebar.columns(2)
    with start_col:
        if start_col.button(
            "▶ Start Camera",
            key="start_camera_button",
            type="primary",
            width="stretch",
            disabled=st.session_state.camera_running,
        ):
            st.session_state.camera_running = True
    with stop_col:
        if stop_col.button(
            "■ Stop Camera",
            key="stop_camera_button",
            width="stretch",
            disabled=not st.session_state.camera_running,
        ):
            st.session_state.camera_running = False

    st.sidebar.caption(
        "Allow browser camera access when prompted. Video is processed continuously "
        "on the server while the camera is running."
    )

    return confidence_threshold, object_filter, st.session_state.camera_running


# ---------------------------------------------------------
# VIDEO AREA
# ---------------------------------------------------------
def create_video_area():
    """
    Add the heading for the live browser camera feed.
    """

    st.subheader("LIVE CAMERA FOOTAGE")
    st.caption("🎥 YOLO detections are drawn directly on the continuous live video.")


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def show_metrics(
    total_detections,
    most_detected,
    confidence_threshold
):
    """
    Display detection statistics.
    """

    col1, col2, col3 = st.columns(3, gap="medium")
    col1.metric(
        "Accepted detections",
        f"{total_detections:,}",
        help="Valid bounding boxes counted across processed video frames.",
    )
    col2.metric("Most detected class", most_detected)
    col3.metric("Confidence threshold", f"{confidence_threshold:.0%}")


def show_welcome(confidence_threshold, object_filter):
    """Show an informative dashboard before the camera starts."""

    st.success("Ready to start", icon="✅")
    show_metrics(0, "—", confidence_threshold)

    col_intro, col_setup = st.columns([1.5, 1], gap="large")
    with col_intro:
        st.subheader("Your live detection workspace")
        st.write(
            "Start the camera when you are ready. The app will annotate matching "
            "objects, update the session statistics, and log events when MySQL is available."
        )
        st.caption(f"Current filter: **{object_filter}** · Minimum confidence: **{confidence_threshold:.0%}**")
    with col_setup:
        st.markdown("#### Quick start")
        st.markdown(
            """
            1. Choose an object class or **All Objects**.
            2. Set the minimum confidence.
            3. Select **Start** in the sidebar.

            Select **Stop** at any time to release the webcam.
            """
        )
    st.divider()


# ---------------------------------------------------------
# RECENT DATABASE LOGS
# ---------------------------------------------------------
def show_recent_logs(logs):
    """
    Display recent MySQL detection logs.
    """

    st.subheader("Recent detection events")

    if not logs:

        st.info(
            "No detection logs available yet."
        )

        return

    try:

        dataframe = pd.DataFrame(logs)

        # Format confidence as percentage
        if "confidence" in dataframe.columns:

            dataframe["confidence"] = (
                dataframe["confidence"] * 100
            ).round(2).astype(str) + "%"
        if "timestamp" in dataframe.columns:
            dataframe["timestamp"] = pd.to_datetime(
                dataframe["timestamp"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
            height=350,
        )

    except Exception as e:

        st.error(
            f"Unable to display logs: {e}"
        )


# ---------------------------------------------------------
# OBJECT STATISTICS
# ---------------------------------------------------------
def show_object_statistics(detected_objects):
    """
    Display number of detections for each object class.
    """

    st.subheader("Session object mix")

    if not detected_objects:

        st.info(
            "No objects detected yet."
        )

        return

    dataframe = pd.DataFrame(
        list(detected_objects.items()),
        columns=[
            "Object",
            "Count"
        ]
    )

    dataframe = dataframe.sort_values(
        by="Count",
        ascending=False
    )

    st.bar_chart(dataframe.set_index("Object"), height=260)


# ---------------------------------------------------------
# INFORMATION PANEL
# ---------------------------------------------------------
def show_information():

    st.subheader("ℹ️ About the Platform")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 🤖 YOLO

            YOLO performs real-time object
            detection using a pre-trained
            computer vision model.
            """
        )

    with col2:

        st.markdown(
            """
            ### 👁️ OpenCV

            OpenCV captures the webcam stream
            and displays the annotated frames.
            """
        )

    with col3:

        st.markdown(
            """
            ### 🗄️ MySQL

            Detection events are stored with
            timestamps, confidence scores,
            and bounding-box coordinates.
            """
        )


# ---------------------------------------------------------
# ERROR MESSAGE
# ---------------------------------------------------------
def show_error(message):

    st.error(
        f"❌ {message}"
    )


# ---------------------------------------------------------
# SUCCESS MESSAGE
# ---------------------------------------------------------
def show_success(message):

    st.success(
        f"✅ {message}"
    )