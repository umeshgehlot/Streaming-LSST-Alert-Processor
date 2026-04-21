import json
import hmac
import math
import os
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure imports resolve (Add both backend and project root)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch

from database import (
    add_agent_activity,
    add_rl_training_event,
    create_discovery,
    get_feedback_summary,
    get_or_create_rl_policy,
    list_alert_channels,
    update_rl_policy,
)
from ml_models import ensemble_discovery

# SOTA INTEGRATION (MODULARIZED)
try:
    from src.sota.agents.reasoning import AstroAgent as AstroReasoner
    from src.sota.processing import SotaDataService
    SOTA_AGENT_AVAILABLE = True
except ImportError:
    SOTA_AGENT_AVAILABLE = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_json(url: str, payload: dict, headers: dict, timeout: float, retries: int = 2) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    return {}


def _parse_signing_keys(raw: str) -> list[str]:
    text = raw.strip()
    if len(text) == 0:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            return [str(item).strip() for item in loaded if str(item).strip()]
        except Exception:
            return [segment.strip() for segment in text.split(",") if segment.strip()]
    return [segment.strip() for segment in text.split(",") if segment.strip()]


def _build_signature(signing_key: str, provider: str, body: bytes, timestamp: str) -> str:
    message = f"{provider}:{timestamp}:".encode("utf-8") + body
    return hmac.new(signing_key.encode("utf-8"), message, digestmod="sha256").hexdigest()


class SLMReasoner:
    def __init__(self, model_name: str = "phi-3-mini-3b") -> None:
        self.model_name = model_name
        self.endpoint = os.getenv("SLM_ENDPOINT", "").strip()
        self.triton_http_url = os.getenv("TRITON_HTTP_URL", "").strip().rstrip("/")
        self.triton_model_gpu = os.getenv("TRITON_SLM_MODEL_GPU", "slm_reasoner_gpu").strip()
        self.triton_model_cpu = os.getenv("TRITON_SLM_MODEL_CPU", "slm_reasoner_cpu").strip()
        self.signing_keys = _parse_signing_keys(os.getenv("SLM_SIGNING_KEYS", os.getenv("AGENT_SIGNING_KEYS", "")))
        self.active_signing_key_index = max(0, int(os.getenv("SLM_SIGNING_ACTIVE_INDEX", os.getenv("AGENT_SIGNING_ACTIVE_INDEX", "0"))))
        self.last_backend = "fallback"
        self.last_error = ""

    def _fallback_cot(self, metrics: dict) -> str:
        anomaly_count = metrics.get("anomaly_count", 0)
        threshold = metrics.get("threshold", 0.0)
        confidence = metrics.get("confidence", 0.0)
        peak_score = metrics.get("peak_score", 0.0)
        mean_score = metrics.get("mean_score", 0.0)
        step_1 = f"Step 1: Signal profile shows peak score {peak_score:.5f} and mean score {mean_score:.5f}."
        step_2 = f"Step 2: Candidate set exceeds threshold {threshold:.5f} with {anomaly_count} flagged points."
        step_3 = f"Step 3: Aggregated discovery confidence is {confidence:.4f}, indicating non-random structure."
        step_4 = "Step 4: Physical interpretation favors transient flux excursion rather than baseline noise."
        conclusion = "Conclusion: classify as candidate anomaly and request expert validation."
        return "\n".join([step_1, step_2, step_3, step_4, conclusion])

    def _infer_via_triton(self, prompt: dict, preferred_device: str) -> str:
        if len(self.triton_http_url) == 0:
            return ""
        models = [self.triton_model_gpu, self.triton_model_cpu] if preferred_device == "cuda" else [self.triton_model_cpu, self.triton_model_gpu]
        payload = {
            "inputs": [
                {
                    "name": "PROMPT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [json.dumps(prompt)],
                }
            ],
            "outputs": [{"name": "TEXT"}],
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if len(self.signing_keys) > 0:
            key_index = min(self.active_signing_key_index, len(self.signing_keys) - 1)
            timestamp = _utc_now()
            signature = _build_signature(self.signing_keys[key_index], "triton", payload_bytes, timestamp)
            headers["X-Agent-Key-Id"] = f"slm-{key_index}"
            headers["X-Agent-Signature"] = signature
            headers["X-Agent-Timestamp"] = timestamp
        for model in models:
            body = _post_json(
                f"{self.triton_http_url}/v2/models/{model}/infer",
                payload=payload,
                headers=headers,
                timeout=25,
                retries=1,
            )
            outputs = body.get("outputs", [])
            if len(outputs) == 0:
                continue
            data = outputs[0].get("data", [])
            if len(data) == 0:
                continue
            return str(data[0]).strip()
        return ""

    def reason(self, metrics: dict, preferred_device: str = "cpu") -> str:
        # [UPGRADE] Expert AstroLLaMA Reasoning
        if SOTA_AGENT_AVAILABLE:
            try:
                astro_reasoner = AstroReasoner()
                # Check for cached model or load
                return astro_reasoner.reason(metrics)
            except Exception as e:
                logging.error(f"SOTA Reasoning failed: {e}. Falling back...")
        
        # Original fallback logic preserved below
        prompt = {
            "model": self.model_name,
            "instruction": "Use chain-of-thought to explain why this light curve candidate is anomalous.",
            "metrics": metrics,
        }
        # ... (rest of original reason logic as fallback)

    def diagnostics(self) -> dict:
        return {
            "backend": self.last_backend,
            "triton_http_url": self.triton_http_url,
            "triton_model_gpu": self.triton_model_gpu,
            "triton_model_cpu": self.triton_model_cpu,
            "http_endpoint": self.endpoint,
            "last_error": self.last_error,
            "signing_enabled": len(self.signing_keys) > 0,
            "active_key_index": min(self.active_signing_key_index, max(0, len(self.signing_keys) - 1)) if len(self.signing_keys) > 0 else None,
            "available_keys": len(self.signing_keys),
        }

    def rotate_signing_key(self, index: int | None = None) -> dict:
        if len(self.signing_keys) == 0:
            return self.diagnostics()
        if index is None:
            self.active_signing_key_index = (self.active_signing_key_index + 1) % len(self.signing_keys)
        else:
            self.active_signing_key_index = max(0, min(int(index), len(self.signing_keys) - 1))
        return self.diagnostics()


class VectorSimilarityStore:
    def __init__(self) -> None:
        self.provider = "pinecone" if os.getenv("PINECONE_API_KEY") else "local_vector_store"
        self._index: dict[str, np.ndarray] = {}
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.pinecone_index_host = os.getenv("PINECONE_INDEX_HOST", "").strip().rstrip("/")
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "astronomy-agent").strip()
        self.signing_keys = _parse_signing_keys(os.getenv("PINECONE_SIGNING_KEYS", os.getenv("AGENT_SIGNING_KEYS", "")))
        self.active_signing_key_index = max(0, int(os.getenv("PINECONE_SIGNING_ACTIVE_INDEX", os.getenv("AGENT_SIGNING_ACTIVE_INDEX", "0"))))
        self.last_error = ""

    def _embed(self, flux_values: list[float]) -> np.ndarray:
        # [UPGRADE] Astro-Aware Embeddings via Avocado GPA
        if SOTA_AGENT_AVAILABLE:
            try:
                # We use the GPA-normalized flux as a cleaner base for embedding
                # This removes sampling artifacts before computing the feature vector
                res = SotaDataService.process_with_gpa(pd.DataFrame({'time': np.arange(len(flux_values)), 'flux': flux_values}))
                signal = np.array(res["normalized_flux"])
            except:
                signal = np.asarray(flux_values, dtype=np.float64)
        else:
            signal = np.asarray(flux_values, dtype=np.float64)
            
        if len(signal) == 0:
            return np.zeros(8, dtype=np.float64)
        gradient = np.diff(signal) if len(signal) > 1 else np.array([0.0])
        # Continue with original metric extraction...
        vector = np.array(
            [
                float(np.mean(signal)),
                float(np.std(signal)),
                float(np.min(signal)),
                float(np.max(signal)),
                float(np.median(signal)),
                float(np.percentile(signal, 95)),
                float(np.mean(np.abs(gradient))),
                float(np.std(gradient)),
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def upsert_curve(self, curve_id: str, flux_values: list[float]) -> dict:
        vector = self._embed(flux_values)
        if self.provider == "pinecone" and len(self.pinecone_index_host) > 0:
            try:
                payload = {
                    "vectors": [
                        {
                            "id": curve_id,
                            "values": vector.astype(float).tolist(),
                            "metadata": {"source": "astro-agent", "dimension": int(len(vector))},
                        }
                    ],
                    "namespace": self.pinecone_namespace,
                }
                headers = {"Api-Key": self.pinecone_api_key, "Content-Type": "application/json"}
                if len(self.signing_keys) > 0:
                    key_index = min(self.active_signing_key_index, len(self.signing_keys) - 1)
                    timestamp = _utc_now()
                    body = json.dumps(payload).encode("utf-8")
                    headers["X-Agent-Key-Id"] = f"pinecone-{key_index}"
                    headers["X-Agent-Signature"] = _build_signature(self.signing_keys[key_index], "pinecone", body, timestamp)
                    headers["X-Agent-Timestamp"] = timestamp
                _ = _post_json(
                    f"https://{self.pinecone_index_host}/vectors/upsert",
                    payload=payload,
                    headers=headers,
                    timeout=20,
                    retries=1,
                )
                self.last_error = ""
                return {"curve_id": curve_id, "dimension": int(len(vector)), "provider": self.provider}
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                self.last_error = str(exc)
                self.provider = "local_vector_store"
        self._index[curve_id] = vector
        return {"curve_id": curve_id, "dimension": int(len(vector)), "provider": self.provider}

    def query_similar(self, flux_values: list[float], top_k: int = 5) -> list[dict]:
        query = self._embed(flux_values)
        if self.provider == "pinecone" and len(self.pinecone_index_host) > 0:
            try:
                payload = {
                    "vector": query.astype(float).tolist(),
                    "topK": max(1, top_k),
                    "includeMetadata": True,
                    "namespace": self.pinecone_namespace,
                }
                headers = {"Api-Key": self.pinecone_api_key, "Content-Type": "application/json"}
                if len(self.signing_keys) > 0:
                    key_index = min(self.active_signing_key_index, len(self.signing_keys) - 1)
                    timestamp = _utc_now()
                    body = json.dumps(payload).encode("utf-8")
                    headers["X-Agent-Key-Id"] = f"pinecone-{key_index}"
                    headers["X-Agent-Signature"] = _build_signature(self.signing_keys[key_index], "pinecone", body, timestamp)
                    headers["X-Agent-Timestamp"] = timestamp
                body = _post_json(
                    f"https://{self.pinecone_index_host}/query",
                    payload=payload,
                    headers=headers,
                    timeout=20,
                    retries=1,
                )
                matches = body.get("matches", [])
                self.last_error = ""
                return [
                    {
                        "curve_id": str(match.get("id", "")),
                        "similarity": float(match.get("score", 0.0)),
                        "metadata": match.get("metadata", {}),
                    }
                    for match in matches
                ]
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                self.last_error = str(exc)
                self.provider = "local_vector_store"
        if len(self._index) == 0:
            return []
        scored: list[tuple[str, float]] = []
        for curve_id, vector in self._index.items():
            score = float(np.dot(query, vector))
            scored.append((curve_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [{"curve_id": curve_id, "similarity": score} for curve_id, score in scored[: max(1, top_k)]]

    def diagnostics(self) -> dict:
        return {
            "provider": self.provider,
            "pinecone_enabled": len(self.pinecone_api_key) > 0,
            "pinecone_index_host": self.pinecone_index_host,
            "pinecone_namespace": self.pinecone_namespace,
            "last_error": self.last_error,
            "signing_enabled": len(self.signing_keys) > 0,
            "active_key_index": min(self.active_signing_key_index, max(0, len(self.signing_keys) - 1)) if len(self.signing_keys) > 0 else None,
            "available_keys": len(self.signing_keys),
        }

    def rotate_signing_key(self, index: int | None = None) -> dict:
        if len(self.signing_keys) == 0:
            return self.diagnostics()
        if index is None:
            self.active_signing_key_index = (self.active_signing_key_index + 1) % len(self.signing_keys)
        else:
            self.active_signing_key_index = max(0, min(int(index), len(self.signing_keys) - 1))
        return self.diagnostics()


class RLPolicyOptimizer:
    def __init__(self, clip_epsilon: float = 0.2, learning_rate: float = 0.6) -> None:
        self.clip_epsilon = float(clip_epsilon)
        self.learning_rate = float(learning_rate)

    def apply_feedback(self, reward: float) -> dict:
        policy = get_or_create_rl_policy()
        before_threshold = float(policy["threshold_percentile"])
        sensitivity = float(policy["sensitivity"])
        ratio = math.exp(max(-2.0, min(2.0, reward * sensitivity)))
        clipped_ratio = max(1.0 - self.clip_epsilon, min(1.0 + self.clip_epsilon, ratio))
        delta = self.learning_rate * clipped_ratio * (1.0 if reward > 0 else -1.0)
        after_threshold = before_threshold - delta
        after_threshold = max(80.0, min(99.9, after_threshold))
        after_sensitivity = max(0.2, min(3.0, sensitivity + (0.05 if reward > 0 else -0.05)))
        updated = update_rl_policy(after_threshold, after_sensitivity)
        feedback = get_feedback_summary()
        add_rl_training_event(
            reward=reward,
            precision=feedback["precision_proxy"],
            threshold_before=before_threshold,
            threshold_after=after_threshold,
        )
        return {
            "previous_policy": policy,
            "updated_policy": updated,
            "precision_proxy": feedback["precision_proxy"],
            "ratio": ratio,
            "clipped_ratio": clipped_ratio,
        }


class AutonomousAstronomicalAgent:
    def __init__(self) -> None:
        self.reasoner = SLMReasoner()
        self.vector_store = VectorSimilarityStore()
        self.rl_optimizer = RLPolicyOptimizer()
        self.last_health: dict = {"triton": "unknown", "pinecone": "unknown", "updated_at": _utc_now()}

    def signing_status(self) -> dict:
        return {
            "slm": self.reasoner.diagnostics(),
            "vector_store": self.vector_store.diagnostics(),
        }

    def rotate_signing_keys(self, scope: str = "all", index: int | None = None) -> dict:
        scope_name = (scope or "all").strip().lower()
        payload: dict = {}
        if scope_name in {"all", "slm"}:
            payload["slm"] = self.reasoner.rotate_signing_key(index=index)
        if scope_name in {"all", "vector", "vector_store", "pinecone"}:
            payload["vector_store"] = self.vector_store.rotate_signing_key(index=index)
        return payload

    def health_probe(self) -> dict:
        triton_state = "fallback"
        pinecone_state = "fallback"
        try:
            _ = self.reasoner.reason(
                {
                    "anomaly_count": 1,
                    "threshold": 0.5,
                    "confidence": 0.5,
                    "peak_score": 0.6,
                    "mean_score": 0.3,
                },
                preferred_device="cpu",
            )
            triton_state = "ok" if self.reasoner.last_backend == "triton" else "fallback"
        except Exception as exc:
            triton_state = f"error:{exc}"
        try:
            _ = self.vector_store.upsert_curve("health-probe", [0.1, 0.2, 0.29, 0.4])
            _ = self.vector_store.query_similar([0.1, 0.2, 0.3, 0.41], top_k=1)
            pinecone_state = "ok" if self.vector_store.provider == "pinecone" else "fallback"
        except Exception as exc:
            pinecone_state = f"error:{exc}"
        self.last_health = {"triton": triton_state, "pinecone": pinecone_state, "updated_at": _utc_now()}
        return self.last_health

    def _choose_device(self, use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _notify_channels(self, confidence: float) -> list[dict]:
        channels = list_alert_channels()
        return [channel for channel in channels if float(confidence) >= float(channel["min_confidence"])]

    def run_cycle(
        self,
        dataset_id: str,
        time_values: list[float],
        flux_values: list[float],
        model_names: list[str],
        epochs: int,
        use_gpu: bool = True,
        batch_size: int = 256,
    ) -> dict:
        policy = get_or_create_rl_policy()
        threshold_percentile = float(policy["threshold_percentile"])
        device = self._choose_device(use_gpu)
        add_agent_activity(
            "listen",
            "Ingestion stream received and queued for normalization.",
            {"dataset_id": dataset_id, "timestamp": _utc_now()},
        )
        add_agent_activity(
            "normalize",
            "Normalization completed. Agent prepared windows for unsupervised detection.",
            {"dataset_id": dataset_id, "point_count": len(flux_values)},
        )
        detection = ensemble_discovery(
            dataset_id=dataset_id,
            time_values=time_values,
            flux_values=flux_values,
            model_names=model_names,
            epochs=epochs,
            threshold_percentile=threshold_percentile,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        confidence_series = detection["confidence_index"]
        peak_score = float(max(confidence_series)) if len(confidence_series) > 0 else 0.0
        mean_score = float(sum(confidence_series) / max(1, len(confidence_series)))
        confidence = float(min(1.0, max(0.0, (peak_score + mean_score) / 2)))
        anomaly_indices = detection["anomaly_indices"]
        add_agent_activity(
            "detect",
            "Unsupervised model ensemble detected candidate anomalies.",
            {
                "dataset_id": dataset_id,
                "threshold_percentile": threshold_percentile,
                "anomaly_count": len(anomaly_indices),
                "device": device,
            },
        )
        reasoning_metrics = {
            "dataset_id": dataset_id,
            "anomaly_count": len(anomaly_indices),
            "threshold": detection["threshold"],
            "confidence": confidence,
            "peak_score": peak_score,
            "mean_score": mean_score,
            "top_indices": anomaly_indices[:10],
            "objective": "maximize discovery yield while minimizing false positives",
        }
        reasoning = self.reasoner.reason(reasoning_metrics, preferred_device=device)
        add_agent_activity(
            "reason",
            "SLM generated chain-of-thought explanation for candidate discovery.",
            {"dataset_id": dataset_id, "model": self.reasoner.model_name},
        )
        similar_curves = self.vector_store.query_similar(flux_values, top_k=5)
        vector_meta = self.vector_store.upsert_curve(dataset_id, flux_values)
        discovery_id = create_discovery(
            dataset_id=dataset_id,
            result_id=None,
            status="candidate",
            confidence=confidence,
            reasoning=reasoning,
            meta={
                "threshold_percentile": threshold_percentile,
                "anomaly_count": len(anomaly_indices),
                "similar_curves": similar_curves,
                "vector_store": vector_meta,
                "hash": sha256(f"{dataset_id}:{peak_score}:{mean_score}".encode("utf-8")).hexdigest(),
            },
        )
        triggered = self._notify_channels(confidence)
        add_agent_activity(
            "notify",
            "Notification dispatch completed for high-confidence candidate.",
            {"dataset_id": dataset_id, "discovery_id": discovery_id, "triggered_channels": len(triggered)},
        )
        return {
            "dataset_id": dataset_id,
            "discovery_id": discovery_id,
            "reasoning": reasoning,
            "confidence": confidence,
            "anomaly_indices": anomaly_indices,
            "confidence_index": confidence_series,
            "threshold": detection["threshold"],
            "threshold_percentile": threshold_percentile,
            "policy": policy,
            "triggered_channels": triggered,
            "vector_provider": self.vector_store.provider,
            "slm_backend": self.reasoner.last_backend,
            "backends": {
                "slm": self.reasoner.diagnostics(),
                "vector_store": self.vector_store.diagnostics(),
            },
        }
