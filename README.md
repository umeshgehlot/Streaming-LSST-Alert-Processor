# Unsupervised Deep Learning for Anomaly Discovery in Astronomical Time-Series Data

## Project Structure

```text
Project/
├─ backend/
│  ├─ __init__.py
│  ├─ data_processing.py
│  ├─ database.py
│  ├─ external_sources.py
│  ├─ main.py
│  ├─ ml_models.py
│  ├─ requirements.txt
│  ├─ results/
│  └─ schemas.py
├─ data/
│  ├─ synthetic_light_curve.csv
│  └─ uploads/
├─ frontend/
│  ├─ eslint.config.js
│  ├─ index.html
│  ├─ package.json
│  ├─ postcss.config.js
│  ├─ tailwind.config.js
│  ├─ vite.config.js
│  └─ src/
│     ├─ api.js
│     ├─ App.jsx
│     ├─ index.css
│     └─ main.jsx
├─ models/
│  └─ .gitkeep
├─ requirements.txt
└─ README.md
```

## Features

- CSV upload for astronomical light curve data (`time`, `flux`)
- NASA/JPL Fireball API ingestion (recent years, converted to time-series)
- Data preprocessing with missing value handling and normalization
- Recent 2-year filtering for datetime-based datasets (configurable in UI/API)
- Deep anomaly detection with:
  - Autoencoder
  - Variational Autoencoder
  - Transformer-based reconstructor
- Interactive visualization:
  - Time-series line plot
  - Highlighted anomalies
  - Anomaly score chart
- Model comparison table
- One-click multi-model comparison endpoint and UI action
- Download anomaly results as CSV
- Configurable anomaly threshold percentile
- Dataset summary statistics endpoint and dashboard cards
- Results history with model filter and pagination
- SQLite-backed local persistence
- `backend/app.db` is created automatically on first backend run

## Backend Setup

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Run Both Together

- Start backend on `http://127.0.0.1:8000`
- Start frontend on `http://127.0.0.1:5173`
- Open the frontend URL in browser

## API Endpoints

- `POST /upload` – Upload CSV and return parsed/normalized points (supports `recent_only`, `recent_years`)
- `GET /fetch/nasa-fireball` – Fetch external NASA/JPL fireball data and ingest as dataset
- `POST /train` – Train selected models for a dataset
- `POST /detect` – Run anomaly detection with one model (supports `threshold_percentile`)
- `POST /compare` – Train+detect across selected models and return comparison output
- `GET /datasets/{dataset_id}/summary` – Get processed stats and time metadata
- `GET /results?dataset_id=<id>` – Fetch historical results with pagination and model filter
- `GET /results/{result_id}/download` – Download score output CSV

## Sample Data

- Use `data/synthetic_light_curve.csv` for a quick local test.

## Local Commands

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm run dev
```
