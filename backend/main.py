import logging
import json
import asyncio
import os
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from auth import authenticate_user, create_access_token, get_current_user
from security import (
    limiter,
    get_rate_limit,
    get_current_user_active,
    SecurityHeaders,
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)
from agentic import AutonomousAstronomicalAgent
from database import (
    add_agent_activity,
    add_alert_channel,
    add_expert_feedback,
    add_collaboration_comment,
    add_provenance_block,
    add_public_label,
    count_results,
    create_result,
    create_collaboration_room,
    create_train_run,
    get_feedback_summary,
    get_gamification_leaderboard,
    get_last_block_hash,
    get_or_create_rl_policy,
    fetch_dataset,
    fetch_results,
    list_alert_channels,
    list_agent_activities,
    list_collaboration_comments,
    list_collaboration_rooms,
    list_discoveries,
    list_provenance_blocks,
    list_public_labels,
    list_rl_training_history,
    init_db,
    insert_dataset,
    update_result_feedback,
    update_discovery_status,
)
from data_processing import load_and_preprocess_csv, lomb_scargle_periodogram
from external_sources import fetch_cross_match_metadata, fetch_live_transient_stream, fetch_nasa_fireball_csv
from ml_models import (
    detect_anomalies,
    ensemble_discovery,
    hyperparameter_search,
    latent_projection_3d,
    train_models,
)
from schemas import (
    AlertChannelRequest,
    AgentRunRequest,
    CollaborationCommentRequest,
    CollaborationRoomRequest,
    CompareRequest,
    CrossMatchRequest,
    DetectRequest,
    DetectResponse,
    EnsembleRequest,
    ExpertFeedbackRequest,
    HyperTuneRequest,
    LatentProjectionRequest,
    LoginRequest,
    LoginResponse,
    ModelBuilderRequest,
    PeriodogramRequest,
    ProvenanceRequest,
    PublicationRequest,
    PublicLabelRequest,
    TrainRequest,
    TrainResponse,
    UploadResponse,
    UserProfile,
    VectorQueryRequest,
    SigningRotateRequest,
    ResultFeedbackRequest,
)
from validators import (
    UploadValidation,
    TrainValidation,
    DetectValidation,
    CrossMatchValidation,
    validate_file_extension,
    validate_file_size,
    validate_csv_content,
)
from error_handlers import setup_exception_handlers


app = FastAPI(title="Astronomical Anomaly Discovery API", version="1.0.0")
logger = logging.getLogger("astronomy_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DATASET_CACHE_LIMIT = 24
dataset_cache: OrderedDict[tuple[str, bool, int], dict] = OrderedDict()
dataset_cache_lock = Lock()
job_store: dict[str, dict] = {}
job_store_lock = Lock()
job_executor = ThreadPoolExecutor(max_workers=2)
agent = AutonomousAstronomicalAgent()
transient_event_buffer: deque[dict] = deque(maxlen=500)
health_monitor_task: asyncio.Task | None = None
report_dir = Path(__file__).resolve().parent / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

PRETRAINED_BENCHMARKS = [
    {"id": "kepler_autoencoder_v1", "mission": "Kepler", "model_name": "autoencoder", "window_size": 32},
    {"id": "tess_vae_v1", "mission": "TESS", "model_name": "vae", "window_size": 32},
    {"id": "kepler_transformer_v1", "mission": "Kepler", "model_name": "transformer", "window_size": 32},
]

# Configure CORS with specific origins
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_URL", "")
]
allowed_origins = [origin for origin in allowed_origins if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"]
)

init_db()
setup_exception_handlers(app)

# Setup rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def _health_monitor_loop() -> None:
    interval = max(10, int(os.getenv("AGENT_HEALTH_MONITOR_INTERVAL", "45")))
    previous = {"triton": "unknown", "pinecone": "unknown"}
    while True:
        status = agent.health_probe()
        changed = status["triton"] != previous["triton"] or status["pinecone"] != previous["pinecone"]
        if changed:
            add_agent_activity(
                "health-monitor",
                "Backend health state changed.",
                {"previous": previous, "current": status},
            )
            previous = {"triton": status["triton"], "pinecone": status["pinecone"]}
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_event() -> None:
    global health_monitor_task
    if os.getenv("AGENT_HEALTH_MONITOR_ENABLED", "true").lower() == "true":
        health_monitor_task = asyncio.create_task(_health_monitor_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global health_monitor_task
    if health_monitor_task is not None:
        health_monitor_task.cancel()
        health_monitor_task = None


def get_processed_dataset(dataset_id: str, recent_only: bool, recent_years: int) -> dict:
    key = (dataset_id, recent_only, recent_years)
    with dataset_cache_lock:
        cached = dataset_cache.get(key)
        if cached is not None:
            dataset_cache.move_to_end(key)
            return cached
    dataset_row = fetch_dataset(dataset_id)
    if dataset_row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        parsed = load_and_preprocess_csv(
            dataset_row["raw_bytes"], recent_only=recent_only, recent_years=recent_years
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with dataset_cache_lock:
        dataset_cache[key] = parsed
        dataset_cache.move_to_end(key)
        while len(dataset_cache) > DATASET_CACHE_LIMIT:
            dataset_cache.popitem(last=False)
    return parsed


def run_train_pipeline(request: TrainRequest) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    train_summary = train_models(
        dataset_id=request.dataset_id,
        flux_values=dataset["normalized_flux"],
        model_names=request.models,
        epochs=request.epochs,
        batch_size=request.batch_size,
        use_gpu=request.use_gpu,
        denoising_method=request.denoising_method,
        denoising_strength=request.denoising_strength,
        auxiliary_values=request.auxiliary_values,
    )
    for item in train_summary:
        create_train_run(
            dataset_id=request.dataset_id,
            model_name=item["model_name"],
            epochs=request.epochs,
            final_loss=item["final_loss"],
            model_path=item["model_path"],
        )
    return {"dataset_id": request.dataset_id, "training": train_summary}


def run_compare_pipeline(request: CompareRequest) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    training = train_models(
        dataset_id=request.dataset_id,
        flux_values=dataset["normalized_flux"],
        model_names=request.models,
        epochs=request.epochs,
        batch_size=request.batch_size,
        use_gpu=request.use_gpu,
        denoising_method=request.denoising_method,
        denoising_strength=request.denoising_strength,
        auxiliary_values=request.auxiliary_values,
    )
    comparisons: list[dict] = []
    for trained in training:
        model_name = trained["model_name"]
        create_train_run(
            dataset_id=request.dataset_id,
            model_name=model_name,
            epochs=request.epochs,
            final_loss=trained["final_loss"],
            model_path=trained["model_path"],
        )
        detected = detect_anomalies(
            dataset_id=request.dataset_id,
            model_name=model_name,
            time_values=dataset["time_values"],
            flux_values=dataset["normalized_flux"],
            epochs=request.epochs,
            threshold_percentile=request.threshold_percentile,
            batch_size=request.batch_size,
            use_gpu=request.use_gpu,
            denoising_method=request.denoising_method,
            denoising_strength=request.denoising_strength,
            auxiliary_values=request.auxiliary_values,
        )
        result_id = create_result(
            dataset_id=request.dataset_id,
            model_name=model_name,
            anomaly_indices=detected["anomaly_indices"],
            scores_path=detected["scores_path"],
            threshold=detected["threshold"],
        )
        comparisons.append(
            {
                "result_id": result_id,
                "model_name": model_name,
                "final_loss": trained["final_loss"],
                "anomaly_count": len(detected["anomaly_indices"]),
                "threshold": detected["threshold"],
                "mean_score": sum(detected["scores"]) / max(1, len(detected["scores"])),
            }
        )
    best_model = min(comparisons, key=lambda item: item["final_loss"])
    return {
        "dataset_id": request.dataset_id,
        "meta": dataset["meta"],
        "comparisons": comparisons,
        "best_model": best_model["model_name"],
    }


def build_insight_summary(dataset_id: str, model_name: str, detection: dict) -> tuple[str, str]:
    anomaly_indices = detection.get("anomaly_indices", [])
    highlighted_points = detection.get("highlighted_points", [])
    threshold = float(detection.get("threshold", 0.0))
    if len(anomaly_indices) == 0:
        return (
            f"No high-confidence anomaly spikes detected for dataset {dataset_id} using {model_name} at threshold {threshold:.6f}.",
            "template",
        )
    strongest = max(highlighted_points, key=lambda item: float(item.get("score", 0.0)), default=None)
    strongest_text = (
        "unknown peak"
        if strongest is None
        else f"T+{float(strongest['time']):.2f}s with score {float(strongest['score']):.4f}"
    )
    fallback = (
        f"{len(anomaly_indices)} anomalies detected by {model_name}; strongest spike at {strongest_text}. "
        f"Threshold: {threshold:.6f}. Candidate rapid transient event."
    )
    slm_endpoint = os.getenv("ASTRO_SLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
    slm_model = os.getenv("ASTRO_SLM_MODEL", "phi3:mini")
    slm_enabled = os.getenv("ASTRO_SLM_ENABLED", "true").lower() == "true"
    if not slm_enabled:
        return fallback, "template"
    prompt = "\n".join(
        [
            "You are an astronomy anomaly summarizer.",
            "Write one concise scientific sentence, no bullets.",
            f"Dataset: {dataset_id}",
            f"Model: {model_name}",
            f"Anomaly count: {len(anomaly_indices)}",
            f"Detection threshold: {threshold:.6f}",
            f"Strongest point: {strongest_text}",
            "Tone: objective and research-oriented.",
        ]
    )
    body = json.dumps({"model": slm_model, "prompt": prompt, "stream": False}).encode("utf-8")
    try:
        request = Request(
            slm_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        summary = str(payload.get("response", "")).strip()
        if summary:
            return summary, f"ollama:{slm_model}"
    except Exception:
        pass
    return fallback, "template"


def submit_job(job_type: str, payload: dict) -> str:
    job_id = str(uuid4())
    with job_store_lock:
        job_store[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def execute():
        with job_store_lock:
            job_store[job_id]["status"] = "running"
        try:
            if job_type == "train":
                result = run_train_pipeline(TrainRequest(**payload))
            else:
                result = run_compare_pipeline(CompareRequest(**payload))
            with job_store_lock:
                job_store[job_id]["status"] = "completed"
                job_store[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                job_store[job_id]["result"] = result
        except Exception as exc:
            with job_store_lock:
                job_store[job_id]["status"] = "failed"
                job_store[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                job_store[job_id]["error"] = str(exc)

    job_executor.submit(execute)
    return job_id


@app.middleware("http")
async def request_logger(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/login", response_model=LoginResponse)
@limiter.limit(get_rate_limit("login"))
def login(request: Request, login_request: LoginRequest) -> LoginResponse:
    user = authenticate_user(login_request.email, login_request.password)
    token = create_access_token(user)
    return LoginResponse(
        access_token=token,
        user=UserProfile(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
        ),
    )


@app.get("/auth/me", response_model=UserProfile)
def auth_me(current_user: dict = Depends(get_current_user)) -> UserProfile:
    return UserProfile(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"],
    )


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(get_rate_limit("upload"))
async def upload(
    request: Request,
    file: UploadFile = File(...),
    recent_only: bool = Query(True),
    recent_years: int = Query(2, ge=1, le=10),
    current_user: dict = Depends(get_current_user_active),
) -> UploadResponse:
    # Validate inputs
    validation = UploadValidation(recent_only=recent_only, recent_years=recent_years)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    validate_file_extension(file.filename, ['.csv'])

    # Read and validate file
    raw_bytes = await file.read()
    validate_file_size(len(raw_bytes), max_size_mb=100)
    validate_csv_content(raw_bytes)

    try:
        dataset = load_and_preprocess_csv(
            raw_bytes,
            recent_only=validation.recent_only,
            recent_years=validation.recent_years
        )
        if dataset["points"] == 0:
            raise HTTPException(status_code=400, detail="No valid data points found in file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(exc)}") from exc
    except Exception as exc:
        logger.error(f"Unexpected error during upload: {str(exc)}")
        raise HTTPException(status_code=500, detail="Internal server error during file processing") from exc

    dataset_id = insert_dataset(file.filename, raw_bytes)
    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        points=dataset["points"],
        normalized_points=dataset["normalized_points"],
        meta=dataset["meta"],
    )


@app.get("/fetch/nasa-fireball", response_model=UploadResponse)
def fetch_nasa_fireball(
    recent_years: int = Query(2, ge=1, le=5),
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    try:
        filename, raw_bytes = fetch_nasa_fireball_csv(recent_years=recent_years)
        dataset = load_and_preprocess_csv(raw_bytes, recent_only=True, recent_years=recent_years)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"External fetch failed: {exc}") from exc
    dataset_id = insert_dataset(filename, raw_bytes)
    return UploadResponse(
        dataset_id=dataset_id,
        filename=filename,
        points=dataset["points"],
        normalized_points=dataset["normalized_points"],
        meta=dataset["meta"],
    )


@app.post("/train", response_model=TrainResponse)
@limiter.limit(get_rate_limit("train"))
def train(request: Request, train_request: TrainRequest, current_user: dict = Depends(get_current_user_active)) -> TrainResponse:
    # Validate request
    validation = TrainValidation(**train_request.model_dump())

    try:
        result = run_train_pipeline(train_request)
        if not result or "training" not in result:
            raise HTTPException(status_code=500, detail="Training pipeline failed to return results")
        return TrainResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Training error: {str(exc)}") from exc
    except Exception as exc:
        logger.error(f"Unexpected error during training: {str(exc)}")
        raise HTTPException(status_code=500, detail="Internal server error during training") from exc


@app.post("/train/async")
def train_async(request: TrainRequest, current_user: dict = Depends(get_current_user)) -> dict:
    job_id = submit_job("train", request.model_dump())
    return {"job_id": job_id, "status": "queued"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest, current_user: dict = Depends(get_current_user)) -> DetectResponse:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    result = detect_anomalies(
        dataset_id=request.dataset_id,
        model_name=request.model_name,
        time_values=dataset["time_values"],
        flux_values=dataset["normalized_flux"],
        epochs=request.epochs,
        threshold_percentile=request.threshold_percentile,
        batch_size=request.batch_size,
        use_gpu=request.use_gpu,
        denoising_method=request.denoising_method,
        denoising_strength=request.denoising_strength,
        auxiliary_values=request.auxiliary_values,
    )
    result_id = create_result(
        dataset_id=request.dataset_id,
        model_name=request.model_name,
        anomaly_indices=result["anomaly_indices"],
        scores_path=result["scores_path"],
        threshold=result["threshold"],
    )
    insight_summary, insight_backend = build_insight_summary(request.dataset_id, request.model_name, result)
    return DetectResponse(
        result_id=result_id,
        dataset_id=request.dataset_id,
        model_name=request.model_name,
        anomaly_indices=result["anomaly_indices"],
        threshold=result["threshold"],
        scores=result["scores"],
        highlighted_points=result["highlighted_points"],
        xai_heatmap=result["xai_heatmap"],
        insight_summary=insight_summary,
        insight_backend=insight_backend,
    )


@app.get("/results")
def get_results(
    dataset_id: str = Query(...),
    model_name: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
) -> dict:
    rows = fetch_results(dataset_id=dataset_id, model_name=model_name, limit=limit, offset=offset)
    total = count_results(dataset_id=dataset_id, model_name=model_name)
    return {"dataset_id": dataset_id, "results": rows, "total": total, "limit": limit, "offset": offset}


@app.get("/datasets/{dataset_id}/summary")
def dataset_summary(
    dataset_id: str,
    recent_only: bool = Query(True),
    recent_years: int = Query(2, ge=1, le=10),
    current_user: dict = Depends(get_current_user),
) -> dict:
    dataset = get_processed_dataset(dataset_id, recent_only, recent_years)
    flux_values = dataset["normalized_flux"]
    point_count = len(flux_values)
    if point_count == 0:
        raise HTTPException(status_code=400, detail="Dataset has no valid data")
    mean_flux = sum(flux_values) / point_count
    min_flux = min(flux_values)
    max_flux = max(flux_values)
    variance = sum((value - mean_flux) ** 2 for value in flux_values) / point_count
    std_flux = variance**0.5
    return {
        "dataset_id": dataset_id,
        "meta": dataset["meta"],
        "stats": {
            "points": point_count,
            "mean_flux": mean_flux,
            "min_flux": min_flux,
            "max_flux": max_flux,
            "std_flux": std_flux,
        },
    }


@app.post("/compare")
def compare_models(request: CompareRequest, current_user: dict = Depends(get_current_user)) -> dict:
    return run_compare_pipeline(request)


@app.post("/compare/async")
def compare_models_async(request: CompareRequest, current_user: dict = Depends(get_current_user)) -> dict:
    job_id = submit_job("compare", request.model_dump())
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    with job_store_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/results/{result_id}/download")
def download_result(result_id: str, current_user: dict = Depends(get_current_user)) -> FileResponse:
    rows = fetch_results(None, result_id=result_id)
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail="Result not found")
    target = rows[0]["scores_path"]
    return FileResponse(path=target, filename=f"{result_id}_scores.csv", media_type="text/csv")


@app.post("/results/{result_id}/feedback")
def label_result_feedback(
    result_id: str,
    request: ResultFeedbackRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    try:
        updated = update_result_feedback(result_id, request.feedback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"result_id": result_id, "feedback": request.feedback.strip().lower(), "status": "updated"}


@app.post("/ensemble/discover")
def ensemble_detect(request: EnsembleRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    result = ensemble_discovery(
        dataset_id=request.dataset_id,
        time_values=dataset["time_values"],
        flux_values=dataset["normalized_flux"],
        model_names=request.models,
        epochs=request.epochs,
        threshold_percentile=request.threshold_percentile,
        batch_size=request.batch_size,
        use_gpu=request.use_gpu,
        auxiliary_values=request.auxiliary_values,
        denoising_method=request.denoising_method,
        denoising_strength=request.denoising_strength,
    )
    return {
        "dataset_id": request.dataset_id,
        "confidence_index": result["confidence_index"],
        "threshold": result["threshold"],
        "anomaly_indices": result["anomaly_indices"],
        "models": result["models"],
    }


@app.post("/tune/hyperparameters")
def tune_hyperparameters(request: HyperTuneRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    result = hyperparameter_search(
        dataset_id=request.dataset_id,
        flux_values=dataset["normalized_flux"],
        model_name=request.model_name,
        epochs=request.epochs,
        trial_count=request.trial_count,
        use_gpu=request.use_gpu,
    )
    return {"dataset_id": request.dataset_id, **result}


@app.post("/analysis/periodogram")
def periodogram(request: PeriodogramRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    return {
        "dataset_id": request.dataset_id,
        **lomb_scargle_periodogram(
            time_values=dataset["time_values"],
            flux_values=dataset["normalized_flux"],
            min_frequency=request.min_frequency,
            max_frequency=request.max_frequency,
            steps=request.steps,
        ),
    }


@app.post("/latent/projection")
def latent_projection(request: LatentProjectionRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    projection = latent_projection_3d(dataset["normalized_flux"], sample_limit=request.sample_limit)
    return {"dataset_id": request.dataset_id, **projection}


@app.get("/transfer/benchmarks")
def list_transfer_benchmarks(current_user: dict = Depends(get_current_user)) -> dict:
    return {"benchmarks": PRETRAINED_BENCHMARKS}


@app.post("/transfer/apply/{benchmark_id}")
def apply_transfer_benchmark(benchmark_id: str, dataset_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    benchmark = next((item for item in PRETRAINED_BENCHMARKS if item["id"] == benchmark_id), None)
    if benchmark is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return {"dataset_id": dataset_id, "applied_benchmark": benchmark}


@app.post("/model-builder/train")
def model_builder_train(request: ModelBuilderRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    layers = [size for size in request.layer_sizes if size > 1]
    models = ["autoencoder"] if len(layers) > 0 else ["autoencoder"]
    result = train_models(
        dataset_id=request.dataset_id,
        flux_values=dataset["normalized_flux"],
        model_names=models,
        epochs=request.epochs,
        batch_size=request.batch_size,
        use_gpu=request.use_gpu,
    )
    return {"dataset_id": request.dataset_id, "layer_sizes": layers, "training": result}


@app.post("/collab/rooms")
def create_room(request: CollaborationRoomRequest, current_user: dict = Depends(get_current_user)) -> dict:
    room_id = create_collaboration_room(request.name, request.dataset_id, current_user["email"])
    return {"room_id": room_id}


@app.get("/collab/rooms")
def get_rooms(dataset_id: str | None = Query(None), current_user: dict = Depends(get_current_user)) -> dict:
    return {"rooms": list_collaboration_rooms(dataset_id)}


@app.post("/collab/comments")
def post_comment(request: CollaborationCommentRequest, current_user: dict = Depends(get_current_user)) -> dict:
    comment_id = add_collaboration_comment(request.room_id, current_user["email"], request.message)
    return {"comment_id": comment_id}


@app.get("/collab/comments")
def get_comments(room_id: str = Query(...), current_user: dict = Depends(get_current_user)) -> dict:
    return {"room_id": room_id, "comments": list_collaboration_comments(room_id)}


@app.get("/streams/live")
def live_stream(limit: int = Query(20, ge=1, le=100), current_user: dict = Depends(get_current_user)) -> dict:
    synthetic = fetch_live_transient_stream(limit=limit)
    latest_live = list(transient_event_buffer)[-limit:]
    return {"events": latest_live + synthetic}


@app.websocket("/ws/transient-ingest")
async def transient_ingest(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            transient_event_buffer.append(
                {
                    "event_id": str(message.get("event_id", uuid4())),
                    "survey": str(message.get("survey", "external")),
                    "ra": float(message.get("ra", 0.0)),
                    "dec": float(message.get("dec", 0.0)),
                    "magnitude": float(message.get("magnitude", 0.0)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            await websocket.send_json({"status": "accepted", "buffer_size": len(transient_event_buffer)})
    except WebSocketDisconnect:
        return


@app.post("/crossmatch")
def cross_match(request: CrossMatchRequest, current_user: dict = Depends(get_current_user)) -> dict:
    return fetch_cross_match_metadata(request.object_name)


@app.post("/publication/report")
def generate_publication_report(request: PublicationRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, recent_only=True, recent_years=2)
    summary = {
        "points": len(dataset["normalized_flux"]),
        "mean": sum(dataset["normalized_flux"]) / max(1, len(dataset["normalized_flux"])),
    }
    latex = "\n".join(
        [
            "\\documentclass{article}",
            "\\begin{document}",
            f"\\section*{{{request.title}}}",
            f"Dataset: {request.dataset_id}\\\\",
            f"Points: {summary['points']}\\\\",
            f"Mean normalized flux: {summary['mean']:.6f}\\\\",
            f"Result: {request.result_id or 'N/A'}",
            "\\end{document}",
        ]
    )
    target = report_dir / f"{request.dataset_id}_{uuid4().hex}.tex"
    target.write_text(latex, encoding="utf-8")
    return {"dataset_id": request.dataset_id, "tex_path": str(target), "latex": latex}


@app.get("/publication/download")
def download_publication(tex_path: str = Query(...), current_user: dict = Depends(get_current_user)) -> FileResponse:
    file_path = Path(tex_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="text/plain")


@app.post("/portal/labels")
def create_public_label(request: PublicLabelRequest, current_user: dict = Depends(get_current_user)) -> dict:
    label_id = add_public_label(
        dataset_id=request.dataset_id,
        point_index=request.point_index,
        label=request.label,
        user_name=request.user_name,
    )
    return {"label_id": label_id}


@app.get("/portal/labels")
def get_public_labels(dataset_id: str = Query(...), limit: int = Query(200, ge=1, le=500)) -> dict:
    return {"dataset_id": dataset_id, "labels": list_public_labels(dataset_id, limit=limit)}


@app.get("/portal/leaderboard")
def portal_leaderboard(limit: int = Query(20, ge=1, le=200)) -> dict:
    return {"leaderboard": get_gamification_leaderboard(limit=limit)}


@app.post("/alerts/channels")
def register_alert_channel(request: AlertChannelRequest, current_user: dict = Depends(get_current_user)) -> dict:
    channel_id = add_alert_channel(request.channel_type, request.target, request.min_confidence)
    return {"channel_id": channel_id}


@app.get("/alerts/channels")
def get_alert_channels(current_user: dict = Depends(get_current_user)) -> dict:
    return {"channels": list_alert_channels()}


@app.post("/alerts/dispatch")
def dispatch_alert(dataset_id: str, confidence: float, result_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    channels = list_alert_channels()
    triggered = [
        channel
        for channel in channels
        if float(confidence) >= float(channel["min_confidence"])
    ]
    return {"dataset_id": dataset_id, "result_id": result_id, "confidence": confidence, "triggered": triggered}


@app.post("/provenance/anchor")
def anchor_provenance(request: ProvenanceRequest, current_user: dict = Depends(get_current_user)) -> dict:
    payload_encoded = json.dumps(request.payload, sort_keys=True).encode("utf-8")
    payload_hash = sha256(payload_encoded).hexdigest()
    previous_hash = get_last_block_hash()
    block_hash = sha256(f"{payload_hash}:{previous_hash or ''}:{request.dataset_id}:{request.result_id}".encode("utf-8")).hexdigest()
    block_id = add_provenance_block(
        dataset_id=request.dataset_id,
        result_id=request.result_id,
        payload_hash=payload_hash,
        previous_hash=previous_hash,
        block_hash=block_hash,
    )
    return {"block_id": block_id, "block_hash": block_hash, "previous_hash": previous_hash}


@app.get("/provenance/ledger")
def provenance_ledger(dataset_id: str | None = Query(None), limit: int = Query(100, ge=1, le=500)) -> dict:
    return {"blocks": list_provenance_blocks(dataset_id=dataset_id, limit=limit)}


@app.get("/mobile/discoveries")
def mobile_discoveries(dataset_id: str = Query(...), current_user: dict = Depends(get_current_user)) -> dict:
    rows = fetch_results(dataset_id=dataset_id, limit=10, offset=0)
    compact = [
        {
            "result_id": row["id"],
            "model_name": row["model_name"],
            "anomaly_count": len(row["anomaly_indices"]),
            "threshold": row["threshold"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"dataset_id": dataset_id, "discoveries": compact}


@app.get("/sdk/spec")
def sdk_spec() -> dict:
    return {
        "python": {
            "class": "AstroSDK",
            "methods": [
                "login",
                "upload_csv",
                "train",
                "detect",
                "ensemble_discover",
                "run_agent_cycle",
                "infra_status",
                "signing_status",
                "rotate_signing",
                "fetch_results",
            ],
        },
        "cli": {
            "entrypoint": "python -m backend.astro_sdk",
            "commands": ["login", "train", "detect", "ensemble", "agent-run", "infra-status", "signing-status", "rotate-signing"],
        },
    }


@app.get("/platform/capabilities")
def platform_capabilities() -> dict:
    return {
        
    }


@app.get("/infra/status")
def infrastructure_status(
    live_probe: bool = Query(False),
    current_user: dict = Depends(get_current_user),
) -> dict:
    status = {
        "triton_http_url": os.getenv("TRITON_HTTP_URL", ""),
        "triton_gpu_model": os.getenv("TRITON_SLM_MODEL_GPU", "slm_reasoner_gpu"),
        "triton_cpu_model": os.getenv("TRITON_SLM_MODEL_CPU", "slm_reasoner_cpu"),
        "pinecone_enabled": len(os.getenv("PINECONE_API_KEY", "")) > 0,
        "pinecone_index_host": os.getenv("PINECONE_INDEX_HOST", ""),
        "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", "astronomy-agent"),
        "agent_backends": {
            "slm": agent.reasoner.diagnostics(),
            "vector_store": agent.vector_store.diagnostics(),
        },
        "health_monitor_enabled": os.getenv("AGENT_HEALTH_MONITOR_ENABLED", "true").lower() == "true",
        "health_monitor_interval": max(10, int(os.getenv("AGENT_HEALTH_MONITOR_INTERVAL", "45"))),
        "health_monitor_running": health_monitor_task is not None and not health_monitor_task.done(),
        "last_health": agent.last_health,
        "live_probe": {"triton": "skipped", "pinecone": "skipped"},
    }
    if live_probe:
        status["live_probe"] = agent.health_probe()
    status["signing"] = agent.signing_status()
    return status


@app.get("/infra/signing/status")
def infra_signing_status(current_user: dict = Depends(get_current_user)) -> dict:
    return {"signing": agent.signing_status()}


@app.post("/infra/signing/rotate")
def infra_signing_rotate(request: SigningRotateRequest, current_user: dict = Depends(get_current_user)) -> dict:
    rotated = agent.rotate_signing_keys(scope=request.scope, index=request.index)
    add_agent_activity(
        "security",
        "Signing key rotation executed.",
        {"scope": request.scope, "index": request.index, "rotated": rotated},
    )
    return {"rotated": rotated}


@app.post("/agent/run-cycle")
def run_agent_cycle(request: AgentRunRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, request.recent_only, request.recent_years)
    return agent.run_cycle(
        dataset_id=request.dataset_id,
        time_values=dataset["time_values"],
        flux_values=dataset["normalized_flux"],
        model_names=request.models,
        epochs=request.epochs,
        use_gpu=request.use_gpu,
        batch_size=request.batch_size,
    )


@app.get("/agent/activity-feed")
def agent_activity_feed(limit: int = Query(100, ge=1, le=500), current_user: dict = Depends(get_current_user)) -> dict:
    return {"activities": list_agent_activities(limit=limit)}


@app.websocket("/ws/agent/activity")
async def agent_activity_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"activities": list_agent_activities(limit=40)})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


@app.get("/discoveries")
def discoveries(dataset_id: str | None = Query(None), limit: int = Query(100, ge=1, le=500), current_user: dict = Depends(get_current_user)) -> dict:
    return {"discoveries": list_discoveries(dataset_id=dataset_id, limit=limit)}


@app.get("/discoveries/summary")
def discoveries_summary(dataset_id: str | None = Query(None), current_user: dict = Depends(get_current_user)) -> dict:
    items = list_discoveries(dataset_id=dataset_id, limit=1000)
    confirmed = sum(1 for item in items if item["status"] == "confirmed")
    candidate = sum(1 for item in items if item["status"] == "candidate")
    rejected = sum(1 for item in items if item["status"] == "rejected")
    return {
        "total": len(items),
        "confirmed": confirmed,
        "candidate": candidate,
        "rejected": rejected,
        "discovery_yield": confirmed / max(1, confirmed + rejected),
    }


@app.post("/discoveries/feedback")
def submit_discovery_feedback(request: ExpertFeedbackRequest, current_user: dict = Depends(get_current_user)) -> dict:
    reward = 1.0 if request.thumbs_up else -1.0
    feedback_id = add_expert_feedback(
        discovery_id=request.discovery_id,
        researcher=current_user["email"],
        reward=reward,
        notes=request.notes,
    )
    update_discovery_status(request.discovery_id, "confirmed" if request.thumbs_up else "rejected")
    policy_update = agent.rl_optimizer.apply_feedback(reward=reward)
    return {"feedback_id": feedback_id, "policy_update": policy_update}


@app.get("/rl/trainer/stats")
def rl_trainer_stats(current_user: dict = Depends(get_current_user)) -> dict:
    return {
        "policy": get_or_create_rl_policy(),
        "feedback": get_feedback_summary(),
        "history": list_rl_training_history(limit=200),
    }


@app.post("/vector/query")
def vector_similarity_query(request: VectorQueryRequest, current_user: dict = Depends(get_current_user)) -> dict:
    dataset = get_processed_dataset(request.dataset_id, recent_only=False, recent_years=2)
    similar = agent.vector_store.query_similar(dataset["normalized_flux"], top_k=request.top_k)
    return {"dataset_id": request.dataset_id, "provider": agent.vector_store.provider, "matches": similar}
