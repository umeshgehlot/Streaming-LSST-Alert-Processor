import mlflow
import os
import logging
from typing import Dict, Any

class ExperimentTracker:
    """
    Expert-level experiment tracking using MLflow.
    """
    @staticmethod
    def start_tracking(experiment_name: str = "SOTA_Discovery_Ensemble"):
        mlflow.set_experiment(experiment_name)

    @staticmethod
    def log_trial(model_name: str, metrics: Dict[str, float], params: Dict[str, Any]):
        with mlflow.start_run(run_name=f"{model_name}_{int(os.getpid())}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            logging.info(f"MLflow Trial: {model_name} logged.")
            
    @staticmethod
    def log_discovery(candidate_id: str, score: float, interpretation: str):
        """Logs a specific discovery event as an artifact."""
        with mlflow.start_run(run_name=f"Discovery_{candidate_id}"):
            mlflow.log_metric("anomaly_score", score)
            # Log the scientific reasoning as a text artifact
            with open("temp_reasoning.txt", "w") as f:
                f.write(interpretation)
            mlflow.log_artifact("temp_reasoning.txt")
            os.remove("temp_reasoning.txt")
