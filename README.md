## Architectural Integration Of Speech Systems with Interactive Dashboard

##  Overview
This project implements a **complete end-to-end speech understanding pipeline** with an interactive **Streamlit dashboard**.

The system allows users to:
- Upload new audio files
- Automatically rename and prepare inputs
- Run the full pipeline (`main.py`)
- Generate transcription, speaker embeddings, prosody alignment, and speech output
- Store outputs in **separate run folders**
- Visualize results (text, audio, images) in a clean UI

---

##  Features

###  Core Pipeline
- Audio preprocessing (resampling, denoising)
- Speech-to-Text using **Whisper (openai/whisper-tiny)**
- Speaker embedding extraction (SpeechBrain x-vector)
- Prosody alignment
- Text-to-Speech generation

---

###  Dashboard Features
- View **current input files**
- View **current output files and images**
- Upload new audio files (auto-renamed)
- Run pipeline from UI
- Each run saved in:

outputs/run_YYYYMMDD_HHMMSS/

- View:
- Transcript
- JSON segments
- Generated audio
- Output images
- All artifacts

---

##  Project Structure


speech_understanding_vscode_project/
│
├── dashboard.py # Streamlit UI
├── main.py # Main pipeline runner
│
├── data/
│ └── raw/
│ ├── lecture.wav
│ └── student_voice_ref.wav
│
├── outputs/
│ ├── transcript.txt
│ ├── transcript_segments.json
│ ├── output_LRL_cloned.wav
│ ├── prosody_alignment.json
│ │
│ ├── run_20260418_153000/
│ ├── run_20260418_160500/
│
├── models/
│ ├── student_xvector.pt
│
├── src/
│ ├── preprocessing/
│ ├── stt/
│ ├── speaker/
│ ├── prosody/
│ ├── tts/
│
├── environment.yml
├── requirements.txt
└── README.md


## ⚙️ Installation

1. Clone Repository
git clone <your-repo-url>
cd Architectural_Integration_Of_Speech_Systems_with_Interactive_Dashboard

2. Create Conda Environment
conda env create -f environment.yml
conda activate speech_understanding

3. Verify Setup
python main.py

## Running the Dashboard
streamlit run dashboard.py

## Open in browser:
http://localhost:8501

## Input Workflow
1. Upload Files in Dashboard
2. Upload any audio file
3. It will be automatically renamed to:
    data/raw/lecture.wav
    data/raw/student_voice_ref.wav
Supported Formats
.wav (recommended)

## Pipeline Execution
Click Run Pipeline in dashboard:
What happens:
Uses current files in data/raw/

Runs main.py
Generates outputs in outputs/

## Creates new folder:
outputs/run_YYYYMMDD_HHMMSS/
Copies outputs into that folder
Displays results in dashboard
## Outputs Generated
Output	Description
transcript.txt	Final transcription
transcript_segments.json	Timestamped segments
output_LRL_cloned.wav	Generated speech
prosody_alignment.json	Alignment metadata
Images (.png)	Visual outputs
## Model Details
Speech-to-Text
Model: openai/whisper-tiny
Fast but lower accuracy
CPU-friendly
## Speaker Embedding
Model: SpeechBrain ECAPA-TDNN
Generates x-vector embeddings
## TTS Output
Generates speech based on transcription + reference voice
## Limitations
Not real-time (batch processing)
Accuracy depends on:
audio quality
background noise
language clarity
Whisper tiny may mis-transcribe some words
Large audio → slower processing
## Tips for Best Results
Use clean .wav audio
Single speaker input
Minimal background noise
16kHz sampling recommended
(Optional) Docker Support

You can containerize the app using:

## Dockerfile
docker-compose.yml


## Future Improvements
Upgrade to Whisper small/medium
GPU acceleration
Real-time streaming
Audio format auto-conversion
Advanced visualization

## Author
Purushothaman S
M25DE1033 - M.Tech Data Engineering 
Indian Institute of Technology Jodhpur