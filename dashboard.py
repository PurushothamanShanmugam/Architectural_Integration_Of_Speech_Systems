from pathlib import Path
import subprocess
import shutil
import json
from datetime import datetime
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_LECTURE = RAW_DIR / "lecture.wav"
CURRENT_STUDENT = RAW_DIR / "student_voice_ref.wav"


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Architectural Integration Of Speech Systems with Interactive Dashboard",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }

    .main {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 50%, #f9fbff 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #111827 !important;
    }

    /* -------- Hero -------- */
    .hero-card {
        background: #1d4ed8 !important;   /* uniform blue */
        border-radius: 26px;
        padding: 30px 32px;
        box-shadow: 0 14px 34px rgba(30, 64, 175, 0.18);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.18);
    }

    .hero-title {
        color: #ffffff !important;
        margin: 0 !important;
        font-size: 2.35rem !important;
        font-weight: 900 !important;
        line-height: 1.2 !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-icon {
        font-size: 2.2rem;
        line-height: 1;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.18));
    }

    .hero-desc {
        margin-top: 0.9rem !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.6 !important;
    }

    /* -------- Clickable nav pills -------- */
    .nav-pills {
        margin-top: 1.1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .nav-pill {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        color: #ffffff !important;
        font-weight: 800;
        text-decoration: none !important;
        border: 1px solid rgba(255,255,255,0.28);
        transition: all 0.2s ease;
        cursor: pointer !important;
        pointer-events: auto !important;
    }

    .nav-pill:hover {
        background: rgba(255,255,255,0.28);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
        color: #ffffff !important;
    }

    .nav-pill:visited,
    .nav-pill:active,
    .nav-pill:focus {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    /* -------- Cards -------- */
    .section-card {
        background: rgba(255,255,255,0.96);
        border: 1.5px solid #dbeafe;
        border-radius: 22px;
        padding: 20px 22px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .metric-card {
        border-radius: 20px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.06);
        color: #111827 !important;
        border: 1.5px solid rgba(255,255,255,0.85);
    }

    .metric-blue {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    }

    .metric-pink {
        background: linear-gradient(135deg, #fce7f3, #fbcfe8);
    }

    .metric-yellow {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
    }

    .metric-green {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 900;
    }

    .metric-sub {
        margin-top: 0.35rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #374151 !important;
    }

    .mono-box {
        background: #f8fafc;
        border: 1.5px solid #cbd5e1;
        border-radius: 14px;
        padding: 12px 14px;
        font-family: monospace;
        color: #111827 !important;
        font-size: 0.92rem;
        white-space: pre-wrap;
    }

    textarea, .stTextArea textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border-radius: 14px !important;
        border: 1.5px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }

    .gallery-caption {
        text-align: center;
        color: #111827 !important;
        font-size: 0.92rem;
        font-weight: 800;
        margin-top: 0.35rem;
        margin-bottom: 0.85rem;
    }

    .small-note {
        color: #374151 !important;
        font-size: 0.94rem;
        font-weight: 700;
    }

    .audio-label {
        font-size: 0.95rem;
        font-weight: 800;
        color: #111827 !important;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    .footer-note {
        text-align: center;
        color: #111827 !important;
        font-size: 0.92rem;
        font-weight: 800;
        margin-top: 1rem;
    }

    .anchor-space {
        position: relative;
        top: -10px;
        visibility: hidden;
    }

    /* Better button look */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 800 !important;
        min-height: 48px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()

def save_uploaded_file(uploaded_file, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def read_text(path: Path, default: str = "") -> str:
    if not file_exists(path):
        return default
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default

def read_json(path: Path):
    if not file_exists(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def get_current_output_files():
    return sorted(
        [p for p in OUTPUT_DIR.iterdir() if p.is_file()],
        key=lambda x: x.name.lower()
    ) if OUTPUT_DIR.exists() else []

def get_current_output_images():
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower()
    ) if OUTPUT_DIR.exists() else []

def get_current_output_audio():
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    return sorted(
        [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower()
    ) if OUTPUT_DIR.exists() else []

def get_run_dirs_inside_outputs():
    return sorted(
        [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda x: x.name,
        reverse=True
    ) if OUTPUT_DIR.exists() else []

def clear_top_level_output_files_only():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in OUTPUT_DIR.iterdir():
        if item.is_file():
            item.unlink()

def create_new_run_folder() -> Path:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def snapshot_outputs_to_run(run_dir: Path):
    for item in OUTPUT_DIR.iterdir():
        if item.resolve() == run_dir.resolve():
            continue
        if item.is_file():
            shutil.copy2(item, run_dir / item.name)

def get_files_in_run(run_dir: Path):
    if not run_dir.exists():
        return []
    return sorted([p for p in run_dir.iterdir() if p.is_file()], key=lambda x: x.name.lower())

def get_images_in_run(run_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not run_dir.exists():
        return []
    return sorted(
        [p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower()
    )

def get_audio_in_run(run_dir: Path):
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    if not run_dir.exists():
        return []
    return sorted(
        [p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower()
    )

def latest_run_dir():
    runs = get_run_dirs_inside_outputs()
    return runs[0] if runs else None


# ---------------------------
# Session state
# ---------------------------
if "selected_run" not in st.session_state:
    lr = latest_run_dir()
    st.session_state["selected_run"] = lr.name if lr else None


# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            <span class="hero-icon">🎙️</span>
            <span>Architectural Integration Of Speech Systems with Interactive Dashboard</span>
        </div>
        <div class="hero-desc">
            View current inputs and outputs, upload new audio files, automatically rename them to the required pipeline filenames,
            run the full workflow, and inspect each run directly from the dashboard.
        </div>
        <div class="nav-pills">
            <a class="nav-pill" href="#section-current-inputs">Current Inputs</a>
            <a class="nav-pill" href="#section-current-outputs">Current Outputs</a>
            <a class="nav-pill" href="#section-upload-rename">Upload your Audio File </a>
            <a class="nav-pill" href="#section-run-pipeline">Run Pipeline</a>
            <a class="nav-pill" href="#section-saved-runs">Saved Runs</a>
            <a class="nav-pill" href="#section-selected-run">Selected Run View</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Metrics
# ---------------------------
current_output_files = get_current_output_files()
current_output_images = get_current_output_images()
current_output_audio = get_current_output_audio()
runs = get_run_dirs_inside_outputs()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">Current Lecture Input</div>
            <div class="metric-value">{'YES' if file_exists(CURRENT_LECTURE) else 'NO'}</div>
            <div class="metric-sub">{'lecture.wav ready' if file_exists(CURRENT_LECTURE) else 'missing lecture.wav'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card metric-pink">
            <div class="metric-label">Current Student Input</div>
            <div class="metric-value">{'YES' if file_exists(CURRENT_STUDENT) else 'NO'}</div>
            <div class="metric-sub">{'student_voice_ref.wav ready' if file_exists(CURRENT_STUDENT) else 'missing student_voice_ref.wav'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card metric-yellow">
            <div class="metric-label">Current Output Files</div>
            <div class="metric-value">{len(current_output_files)}</div>
            <div class="metric-sub">top-level files in outputs/</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <div class="metric-label">Saved Runs</div>
            <div class="metric-value">{len(runs)}</div>
            <div class="metric-sub">run folders inside outputs/</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Current input files
# ---------------------------
st.markdown('<div id="section-current-inputs" class="anchor-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Current Input Files")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Current Lecture File")
    if file_exists(CURRENT_LECTURE):
        st.success("lecture.wav is available")
        st.code(str(CURRENT_LECTURE))
        try:
            st.audio(CURRENT_LECTURE.read_bytes(), format="audio/wav")
        except Exception:
            pass
    else:
        st.error("lecture.wav is missing")
        st.code(str(CURRENT_LECTURE))

with c2:
    st.markdown("#### Current Student Voice File")
    if file_exists(CURRENT_STUDENT):
        st.success("student_voice_ref.wav is available")
        st.code(str(CURRENT_STUDENT))
        try:
            st.audio(CURRENT_STUDENT.read_bytes(), format="audio/wav")
        except Exception:
            pass
    else:
        st.error("student_voice_ref.wav is missing")
        st.code(str(CURRENT_STUDENT))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Current outputs
# ---------------------------
st.markdown('<div id="section-current-outputs" class="anchor-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Current Output Files, Images, and Audio")

co1, co2 = st.columns([1, 1.2])

with co1:
    st.markdown("#### Current Output Files")
    if current_output_files:
        for f in current_output_files:
            size_kb = f.stat().st_size / 1024
            st.write(f"📄 **{f.name}** — {size_kb:.1f} KB")
    else:
        st.info("No current top-level output files found in outputs/")

    st.markdown("#### Current Output Audio")
    if current_output_audio:
        for audio_file in current_output_audio:
            st.markdown(f"<div class='audio-label'>{audio_file.name}</div>", unsafe_allow_html=True)
            try:
                suffix = audio_file.suffix.lower()
                if suffix == ".wav":
                    st.audio(audio_file.read_bytes(), format="audio/wav")
                elif suffix == ".mp3":
                    st.audio(audio_file.read_bytes(), format="audio/mp3")
                elif suffix == ".m4a":
                    st.audio(audio_file.read_bytes(), format="audio/mp4")
                else:
                    st.audio(audio_file.read_bytes())
            except Exception:
                st.warning(f"Could not preview audio: {audio_file.name}")
    else:
        st.info("No current output audio found in outputs/")

with co2:
    st.markdown("#### Current Output Images")
    if current_output_images:
        cols = st.columns(2)
        for idx, img in enumerate(current_output_images):
            with cols[idx % 2]:
                st.image(str(img), use_container_width=True)
                st.markdown(
                    f"<div class='gallery-caption'>{img.name}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No current output images found in outputs/")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Upload interaction
# ---------------------------
st.markdown('<div id="section-upload-rename" class="anchor-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Upload New Audio Files")

st.markdown(
    """
    <div class="small-note">
    Upload any new audio files here. Their original names do not matter.
    They will automatically be saved with the required pipeline names inside <b>data/raw/</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

u1, u2 = st.columns(2)

with u1:
    st.markdown("#### Upload New Lecture Audio")
    uploaded_lecture = st.file_uploader(
        "Choose a lecture audio file",
        type=["wav", "mp3", "m4a", "flac"],
        key="upload_lecture_new",
    )
    if uploaded_lecture is not None:
        save_uploaded_file(uploaded_lecture, CURRENT_LECTURE)
        st.success(f"Uploaded '{uploaded_lecture.name}' and renamed to 'lecture.wav'")
        st.code(str(CURRENT_LECTURE))

with u2:
    st.markdown("#### Upload New Student Voice Audio")
    uploaded_student = st.file_uploader(
        "Choose a student reference audio file",
        type=["wav", "mp3", "m4a", "flac"],
        key="upload_student_new",
    )
    if uploaded_student is not None:
        save_uploaded_file(uploaded_student, CURRENT_STUDENT)
        st.success(f"Uploaded '{uploaded_student.name}' and renamed to 'student_voice_ref.wav'")
        st.code(str(CURRENT_STUDENT))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Run pipeline
# ---------------------------
st.markdown('<div id="section-run-pipeline" class="anchor-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Run Pipeline")

st.markdown(
    """
    <div class="mono-box">
Run behavior:
1. Uses the current files inside data/raw/
2. Calls main.py
3. Refreshes current top-level files in outputs/
4. Creates a new folder inside outputs/ such as outputs/run_YYYYMMDD_HHMMSS/
5. Copies this run's output files into that new folder
6. Displays the new output files, images, and audio in the dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("▶ Run main.py", type="primary", use_container_width=True):
    if not file_exists(CURRENT_LECTURE):
        st.error("Current lecture.wav is missing. Upload it first.")
    elif not file_exists(CURRENT_STUDENT):
        st.error("Current student_voice_ref.wav is missing. Upload it first.")
    else:
        clear_top_level_output_files_only()

        with st.spinner("Running main.py ... please wait while the pipeline executes."):
            result = subprocess.run(
                ["python", "main.py"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

        if result.returncode == 0:
            run_dir = create_new_run_folder()
            snapshot_outputs_to_run(run_dir)
            st.session_state["selected_run"] = run_dir.name
            st.success(f"Pipeline completed successfully. New run saved in: {run_dir}")
            st.text_area("Pipeline Log", result.stdout, height=300)
        else:
            st.error("Pipeline execution failed.")
            st.text_area("Standard Output", result.stdout, height=220)
            st.text_area("Error Output", result.stderr, height=320)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Run selector
# ---------------------------
runs = get_run_dirs_inside_outputs()
run_names = [r.name for r in runs]

st.markdown('<div id="section-saved-runs" class="anchor-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Saved Run Outputs")

if run_names:
    default_index = 0
    if st.session_state.get("selected_run") in run_names:
        default_index = run_names.index(st.session_state["selected_run"])

    selected_run_name = st.selectbox(
        "Choose a run folder inside outputs/",
        options=run_names,
        index=default_index,
    )
    st.session_state["selected_run"] = selected_run_name
else:
    selected_run_name = None
    st.info("No run folders found yet inside outputs/")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Selected run display
# ---------------------------
st.markdown('<div id="section-selected-run" class="anchor-space"></div>', unsafe_allow_html=True)

if selected_run_name:
    selected_run_dir = OUTPUT_DIR / selected_run_name

    run_files = get_files_in_run(selected_run_dir)
    run_images = get_images_in_run(selected_run_dir)
    run_audio = get_audio_in_run(selected_run_dir)

    transcript_path = selected_run_dir / "transcript.txt"
    segments_path = selected_run_dir / "transcript_segments.json"

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"Viewing Run: {selected_run_name}")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.write(f"Transcript: {'✅' if file_exists(transcript_path) else '❌'}")
    with r2:
        st.write(f"Segments JSON: {'✅' if file_exists(segments_path) else '❌'}")
    with r3:
        st.write(f"Generated Audio: {'✅' if len(run_audio) > 0 else '❌'}")
    with r4:
        st.write(f"Images: {'✅' if len(run_images) > 0 else '❌'}")

    st.markdown("</div>", unsafe_allow_html=True)

    a1, a2 = st.columns([1.2, 1])

    with a1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Transcript")
        transcript_text = read_text(transcript_path, "")
        if transcript_text:
            st.text_area("Transcript Output", transcript_text, height=280)
        else:
            st.info("Transcript file not found in this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Run Output Audio")
        if run_audio:
            for audio_file in run_audio:
                st.markdown(f"<div class='audio-label'>{audio_file.name}</div>", unsafe_allow_html=True)
                try:
                    suffix = audio_file.suffix.lower()
                    if suffix == ".wav":
                        st.audio(audio_file.read_bytes(), format="audio/wav")
                    elif suffix == ".mp3":
                        st.audio(audio_file.read_bytes(), format="audio/mp3")
                    elif suffix == ".m4a":
                        st.audio(audio_file.read_bytes(), format="audio/mp4")
                    else:
                        st.audio(audio_file.read_bytes())
                except Exception:
                    st.warning(f"Could not preview audio: {audio_file.name}")
        else:
            st.info("No output audio file found in this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1.05, 1.2])

    with b1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Transcript Segments JSON")
        seg_data = read_json(segments_path)
        if seg_data is not None:
            st.json(seg_data[:10] if isinstance(seg_data, list) else seg_data)
        else:
            st.info("Segments JSON not found in this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Files in This Run")
        if run_files:
            for f in run_files:
                size_kb = f.stat().st_size / 1024
                st.write(f"📄 **{f.name}** — {size_kb:.1f} KB")
        else:
            st.info("No files found in this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Output Images for Selected Run")
    if run_images:
        cols = st.columns(3)
        for idx, img in enumerate(run_images):
            with cols[idx % 3]:
                st.image(str(img), use_container_width=True)
                st.markdown(
                    f"<div class='gallery-caption'>{img.name}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No output images found in this selected run.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div class="footer-note">
        Architectural Integration Of Speech Systems with Interactive Dashboard • current inputs shown first • current outputs and current output audio shown first • uploads renamed automatically • each run saved inside outputs/run_...
    </div>
    """,
    unsafe_allow_html=True,
)