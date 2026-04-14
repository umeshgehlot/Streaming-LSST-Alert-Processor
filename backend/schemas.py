from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str
    role: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


class Point(BaseModel):
    time: float
    flux: float


class DatasetMeta(BaseModel):
    original_points: int
    processed_points: int
    recent_only: bool
    recent_years: int
    time_mode: str
    start_time: str | None = None
    end_time: str | None = None


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    points: list[Point]
    normalized_points: list[Point]
    meta: DatasetMeta


class TrainRequest(BaseModel):
    dataset_id: str
    models: list[str] = Field(default_factory=lambda: ["autoencoder", "vae", "transformer"])
    epochs: int = 20
    recent_only: bool = True
    recent_years: int = 2
    batch_size: int = 256
    use_gpu: bool = True
    denoising_method: str = "none"
    denoising_strength: int = 5
    auxiliary_values: list[float] = Field(default_factory=list)


class TrainingItem(BaseModel):
    model_name: str
    final_loss: float
    model_path: str


class TrainResponse(BaseModel):
    dataset_id: str
    training: list[TrainingItem]


class DetectRequest(BaseModel):
    dataset_id: str
    model_name: str
    epochs: int = 20
    threshold_percentile: float = 95.0
    recent_only: bool = True
    recent_years: int = 2
    batch_size: int = 256
    use_gpu: bool = True
    denoising_method: str = "none"
    denoising_strength: int = 5
    auxiliary_values: list[float] = Field(default_factory=list)


class HighlightedPoint(BaseModel):
    time: float
    flux: float
    score: float


class DetectResponse(BaseModel):
    result_id: str
    dataset_id: str
    model_name: str
    anomaly_indices: list[int]
    threshold: float
    scores: list[float]
    highlighted_points: list[HighlightedPoint]
    xai_heatmap: list[float] = Field(default_factory=list)
    insight_summary: str = ""
    insight_backend: str = "template"


class CompareRequest(BaseModel):
    dataset_id: str
    models: list[str] = Field(default_factory=lambda: ["autoencoder", "vae", "transformer"])
    epochs: int = 20
    threshold_percentile: float = 95.0
    recent_only: bool = True
    recent_years: int = 2
    batch_size: int = 256
    use_gpu: bool = True
    denoising_method: str = "none"
    denoising_strength: int = 5
    auxiliary_values: list[float] = Field(default_factory=list)


class EnsembleRequest(BaseModel):
    dataset_id: str
    models: list[str] = Field(default_factory=lambda: ["autoencoder", "vae", "transformer"])
    epochs: int = 20
    threshold_percentile: float = 95.0
    recent_only: bool = True
    recent_years: int = 2
    batch_size: int = 256
    use_gpu: bool = True
    denoising_method: str = "none"
    denoising_strength: int = 5
    auxiliary_values: list[float] = Field(default_factory=list)


class HyperTuneRequest(BaseModel):
    dataset_id: str
    model_name: str = "autoencoder"
    epochs: int = 12
    trial_count: int = 8
    recent_only: bool = True
    recent_years: int = 2
    use_gpu: bool = True


class PeriodogramRequest(BaseModel):
    dataset_id: str
    recent_only: bool = True
    recent_years: int = 2
    min_frequency: float = 0.01
    max_frequency: float = 2.0
    steps: int = 300


class LatentProjectionRequest(BaseModel):
    dataset_id: str
    recent_only: bool = True
    recent_years: int = 2
    sample_limit: int = 1200


class ModelBuilderRequest(BaseModel):
    dataset_id: str
    layer_sizes: list[int] = Field(default_factory=lambda: [128, 64, 16])
    epochs: int = 20
    recent_only: bool = True
    recent_years: int = 2
    batch_size: int = 256
    use_gpu: bool = True


class CollaborationRoomRequest(BaseModel):
    name: str
    dataset_id: str | None = None


class CollaborationCommentRequest(BaseModel):
    room_id: str
    message: str


class PublicLabelRequest(BaseModel):
    dataset_id: str
    point_index: int
    label: str
    user_name: str


class AlertChannelRequest(BaseModel):
    channel_type: str
    target: str
    min_confidence: float = 0.8


class PublicationRequest(BaseModel):
    dataset_id: str
    result_id: str | None = None
    title: str = "Astronomical Anomaly Report"


class CrossMatchRequest(BaseModel):
    object_name: str


class ProvenanceRequest(BaseModel):
    dataset_id: str
    result_id: str
    payload: dict


class AgentRunRequest(BaseModel):
    dataset_id: str
    models: list[str] = Field(default_factory=lambda: ["autoencoder", "vae", "transformer"])
    epochs: int = 3
    recent_only: bool = True
    recent_years: int = 2
    use_gpu: bool = True
    batch_size: int = 256


class ExpertFeedbackRequest(BaseModel):
    discovery_id: str
    thumbs_up: bool
    notes: str = ""


class VectorQueryRequest(BaseModel):
    dataset_id: str
    top_k: int = 5


class SigningRotateRequest(BaseModel):
    scope: str = "all"
    index: int | None = None


class ResultFeedbackRequest(BaseModel):
    feedback: str
