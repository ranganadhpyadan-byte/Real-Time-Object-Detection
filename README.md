# Real-Time Object Detection & Logging Platform

A local object-detection dashboard built with Python, OpenCV, Ultralytics YOLO, Streamlit, and MySQL. The dashboard supports a webcam feed, class and confidence filters, live metrics, and optional detection-event logging.

## Features

- Live webcam detection with YOLO
- Adjustable confidence threshold and object-class filter
- Live detection counts and recent MySQL events
- MySQL logging with bounding-box coordinates
- Detection continues when MySQL is unavailable; database events are logged when a connection is available
- Standalone OpenCV window mode (`live_detection.py`)

## Requirements

- Python 3.10 or later
- A webcam for live camera mode
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

The dashboard opens with the webcam and continuous YOLO detection already running. Use **Stop Camera** in the sidebar to stop detection and release the webcam; **Start Camera** restarts it. Frames continue to display when no objects are detected. The YOLO model configured by `MODEL_PATH` is downloaded by Ultralytics the first time if it is not already present.

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
| `CAMERA_INDEX` | Webcam device index | `0` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Requested camera dimensions | `640` / `480` |
| `LOG_COOLDOWN` | Minimum seconds between logged events of the same class | `2.0` |

## Project layout

- `main.py` — Streamlit dashboard and live processing
- `detector.py` — YOLO inference wrapper
- `database.py` — MySQL connection and event queries
- `ui.py` — Reusable dashboard components
- `config.py` — Environment-backed application settings
- `live_detection.py` — Standalone OpenCV webcam application
- `database_schema.sql` — Database and detection table schema
