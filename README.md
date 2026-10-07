# Real-Time Object Detection & Logging Platform

The Streamlit application uses the visitor's browser webcam through
`streamlit-webrtc`. It returns a continuous live video stream with YOLO
annotations; it does not take or upload photos, and the deployed Streamlit
server does not try to access a webcam with `cv2.VideoCapture(0)`.

## Run locally

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run main.py
```

Open the local URL printed by Streamlit, select **Start Camera**, and grant
camera permission in the browser. Camera access requires HTTPS in production;
`localhost` is allowed for local development.

The first camera start loads `yolov8n.pt` once. The webcam stream remains
continuous, while inference is capped at 7 frames per second by default. Frames
are reduced to a maximum 640-pixel side before inference and YOLO uses a 416
image size on CPU. PyTorch's CPU thread count is limited to one to avoid
Ultralytics resetting it to all available host threads on each prediction.
Adjust `INFERENCE_FPS`, `INFERENCE_MAX_SIDE`, and `INFERENCE_IMAGE_SIZE` in
local environment variables or Streamlit secrets if needed.

## Streamlit Community Cloud

Deploy the repository using `main.py` as the app entry point. Add MySQL
connection values as Community Cloud app secrets (`DB_HOST`, `DB_USER`,
`DB_PASSWORD`, `DB_NAME`, and `DB_PORT`); never commit credentials or upload a
real `.env` file. `vision_platform.detection_logs` is created when the app
connects if it does not already exist. Keep `LOG_COOLDOWN=2.0` or higher to
limit each object class to at most one insert per cooldown interval.

The camera video is negotiated from the browser using WebRTC and processed on
the Streamlit server. Networks that block direct WebRTC traffic may require a
TURN relay. If needed, provide `TURN_SERVER_URLS` (comma-separated URLs),
`TURN_SERVER_USERNAME`, and `TURN_SERVER_CREDENTIAL` through Streamlit secrets.
Use TURN credentials intended for client-side WebRTC use. The app must not
report the camera as connected until WebRTC is playing and frames are arriving.

## Modules

- `main.py` coordinates Streamlit, the browser video stream, and UI state.
- `detector.py` loads and runs the cached YOLO nano model.
- `database.py` manages MySQL and queues detection events away from video
  processing.
- `ui.py` contains the Streamlit controls and history display.
- `config.py` reads environment variables and Streamlit secrets.
- `live_detection.py` is an optional local OpenCV-camera command-line utility;
  it is not used by the deployed Streamlit app.

The existing Next.js files in `app/`, `components/`, and the root package
configuration are retained, but the Streamlit Community Cloud deployment uses
`main.py` and `requirements.txt`.
