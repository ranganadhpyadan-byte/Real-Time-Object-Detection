# Real-Time Object Detection & Logging Platform

An object-detection dashboard built with Python, OpenCV, Ultralytics YOLO, Streamlit, and MySQL. The dashboard streams the user's browser webcam through WebRTC, applies YOLO continuously, draws detections on the live video, and optionally logs detection events to MySQL.

## Features

- Continuous browser webcam video with YOLO detection overlays
- Adjustable confidence threshold and object-class filter
- Recent MySQL detection events
- MySQL logging with bounding-box coordinates
- Detection continues when MySQL is unavailable; database events are logged when a connection is available
- Standalone OpenCV window mode (`live_detection.py`) for local desktop use

## Requirements

- Python 3.10 or later
- A webcam and browser permission for live camera mode
- MySQL Server for persistent event logging

## Setup on Windows

Open PowerShell in the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your local MySQL connection settings. Do not commit `.env`.

Create the database by running `database_schema.sql` in MySQL Workbench or with the MySQL command-line client from PowerShell:

```powershell
Get-Content .\database_schema.sql | mysql -u root -p
```

The application creates the `detection_logs` table and its indexes if they are missing. Start the Streamlit dashboard:

```powershell
python -m streamlit run main.py
```

The dashboard requests the browser's webcam and starts continuous YOLO detection. Allow camera access when prompted. Use **Stop Camera** in the sidebar to stop the stream and **Start Camera** to resume it. Frames continue to display when no objects are detected. The YOLO model configured by `MODEL_PATH` is downloaded by Ultralytics the first time if it is not already present.

The Streamlit dashboard uses WebRTC because a cloud server cannot access a webcam attached to your computer. Video is streamed from the browser to the app over WebRTC; YOLO inference and MySQL logging run on the app server. Do not use `cv2.VideoCapture(0)` for a deployed dashboard.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository, select the `main` branch, and set the app file to `main.py`.
3. Configure the app's Python version as **3.11** in its advanced settings.
4. If using MySQL, add the following keys in the app's **Settings → Secrets**. Use a MySQL server reachable from the cloud app; `localhost` refers to the cloud container, not your computer.

   ```toml
   DB_HOST = "your-mysql-host"
   DB_USER = "your-mysql-user"
   DB_PASSWORD = "your-mysql-password"
   DB_NAME = "vision_platform"
   DB_PORT = "3306"
   ```

   Detection still runs when MySQL is unavailable. Keep credentials in Streamlit Secrets, not in source files or GitHub.
5. Open the deployed HTTPS URL and allow webcam access in the browser. If the network blocks WebRTC traffic, try another network that permits browser camera streaming.

The repository's `.gitignore` excludes `.env` and downloaded YOLO weights. Ultralytics downloads the configured model when the app first starts.

The included `requirements.txt` uses headless OpenCV for cloud hosting. To use the optional local `live_detection.py` window, install desktop OpenCV in the local environment instead:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python
```

To run the standalone OpenCV window instead:

```powershell
python live_detection.py --camera 0 --confidence 0.70 --object "All Objects"
```

Press `q` in the OpenCV window to stop. The webcam is released when the script exits.

## Configuration

Copy `.env.example` to `.env` and set the values for your machine:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DB_HOST` | MySQL host | `localhost` |
| `DB_USER` | MySQL user | `root` |
| `DB_PASSWORD` | MySQL password | empty |
| `DB_NAME` | Database name | `vision_platform` |
| `DB_PORT` | MySQL port | `3306` |
| `MODEL_PATH` | Ultralytics model path/name | `yolov8n.pt` |
| `DEFAULT_CONFIDENCE` | Initial confidence threshold | `0.70` |
| `LOG_COOLDOWN` | Minimum seconds between logged events of the same class | `2.0` |

## Project layout

- `main.py` — Streamlit dashboard and live processing
- `detector.py` — YOLO inference wrapper
- `database.py` — MySQL connection and event queries
- `ui.py` — Reusable dashboard components
- `config.py` — Environment-backed application settings
- `live_detection.py` — Standalone OpenCV webcam application
- `database_schema.sql` — Database and detection table schema
