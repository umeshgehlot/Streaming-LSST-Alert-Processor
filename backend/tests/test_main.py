import pytest
from fastapi.testclient import TestClient
from database import fetch_dataset


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_endpoint_success(client: TestClient, sample_csv_file: str, auth_headers: dict):
    with open(sample_csv_file, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("test.csv", f, "text/csv")},
            params={"recent_only": True, "recent_years": 2},
            headers=auth_headers
        )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["filename"] == "test.csv"
    assert data["points"] > 0
    assert "meta" in data


def test_upload_invalid_file_type(client: TestClient, auth_headers: dict):
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]


def test_dataset_summary_endpoint(client: TestClient, sample_csv_file: str, auth_headers: dict):
    # First upload a dataset
    with open(sample_csv_file, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("test.csv", f, "text/csv")},
            headers=auth_headers
        )
    dataset_id = upload_response.json()["dataset_id"]

    # Then get summary
    response = client.get(f"/datasets/{dataset_id}/summary", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == dataset_id
    assert "stats" in data
    assert "points" in data["stats"]
    assert "mean_flux" in data["stats"]


def test_dataset_not_found(client: TestClient, auth_headers: dict):
    response = client.get("/datasets/nonexistent/summary", headers=auth_headers)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]


def test_train_endpoint_missing_dataset(client: TestClient, auth_headers: dict):
    train_data = {
        "dataset_id": "nonexistent",
        "models": ["autoencoder"],
        "epochs": 10,
        "batch_size": 32,
        "use_gpu": False,
        "recent_only": True,
        "recent_years": 2
    }
    response = client.post("/train", json=train_data, headers=auth_headers)
    assert response.status_code == 404


def test_detect_endpoint_invalid_request(client: TestClient, auth_headers: dict):
    detect_data = {
        "dataset_id": "nonexistent",
        "model_name": "autoencoder",
        "epochs": 10,
        "threshold_percentile": 95,
        "batch_size": 32,
        "use_gpu": False,
        "recent_only": True,
        "recent_years": 2
    }
    response = client.post("/detect", json=detect_data, headers=auth_headers)
    assert response.status_code == 404


def test_compare_endpoint_invalid_threshold(client: TestClient, auth_headers: dict):
    compare_data = {
        "dataset_id": "nonexistent",
        "models": ["autoencoder"],
        "epochs": 10,
        "threshold_percentile": 150,  # Invalid percentile
        "batch_size": 32,
        "use_gpu": False,
        "recent_only": True,
        "recent_years": 2
    }
    response = client.post("/compare", json=compare_data, headers=auth_headers)
    assert response.status_code == 404


def test_results_endpoint(client: TestClient, auth_headers: dict):
    response = client.get(
        "/results?dataset_id=test&limit=10&offset=0",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert "results" in data
    assert "total" in data


def test_unauthorized_access(client: TestClient):
    response = client.get("/datasets/test/summary")
    # Should redirect or return 401/403 depending on auth implementation
    assert response.status_code in [401, 403, 307]