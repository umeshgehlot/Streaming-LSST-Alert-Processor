import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


def _http_request(url: str, method: str = "GET", payload: dict | None = None, token: str | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url=url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class AstroSDK:
    base_url: str = "http://127.0.0.1:8000"
    token: str | None = None

    def login(self, email: str, password: str) -> dict:
        result = _http_request(
            f"{self.base_url}/auth/login",
            method="POST",
            payload={"email": email, "password": password},
        )
        self.token = result.get("access_token")
        return result

    def train(self, dataset_id: str, models: list[str], epochs: int = 20) -> dict:
        return _http_request(
            f"{self.base_url}/train",
            method="POST",
            payload={"dataset_id": dataset_id, "models": models, "epochs": epochs},
            token=self.token,
        )

    def detect(self, dataset_id: str, model_name: str, threshold_percentile: float = 95.0) -> dict:
        return _http_request(
            f"{self.base_url}/detect",
            method="POST",
            payload={
                "dataset_id": dataset_id,
                "model_name": model_name,
                "threshold_percentile": threshold_percentile,
            },
            token=self.token,
        )

    def ensemble_discover(self, dataset_id: str, models: list[str], threshold_percentile: float = 95.0) -> dict:
        return _http_request(
            f"{self.base_url}/ensemble/discover",
            method="POST",
            payload={
                "dataset_id": dataset_id,
                "models": models,
                "threshold_percentile": threshold_percentile,
            },
            token=self.token,
        )

    def run_agent_cycle(self, dataset_id: str, models: list[str], epochs: int = 3, use_gpu: bool = True) -> dict:
        return _http_request(
            f"{self.base_url}/agent/run-cycle",
            method="POST",
            payload={
                "dataset_id": dataset_id,
                "models": models,
                "epochs": epochs,
                "use_gpu": use_gpu,
            },
            token=self.token,
        )

    def infra_status(self, live_probe: bool = False) -> dict:
        probe = "true" if live_probe else "false"
        return _http_request(f"{self.base_url}/infra/status?live_probe={probe}", token=self.token)

    def signing_status(self) -> dict:
        return _http_request(f"{self.base_url}/infra/signing/status", token=self.token)

    def rotate_signing(self, scope: str = "all", index: int | None = None) -> dict:
        return _http_request(
            f"{self.base_url}/infra/signing/rotate",
            method="POST",
            payload={"scope": scope, "index": index},
            token=self.token,
        )

    def fetch_results(self, dataset_id: str) -> dict:
        return _http_request(f"{self.base_url}/results?dataset_id={dataset_id}", token=self.token)


def main() -> None:
    parser = argparse.ArgumentParser(prog="astro-sdk-cli")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", default=str(Path.home() / ".astro_token"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("--email", required=True)
    login_parser.add_argument("--password", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-id", required=True)
    train_parser.add_argument("--models", nargs="+", default=["autoencoder", "vae", "transformer"])
    train_parser.add_argument("--epochs", type=int, default=20)

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("--dataset-id", required=True)
    detect_parser.add_argument("--model-name", required=True)
    detect_parser.add_argument("--threshold-percentile", type=float, default=95.0)

    ensemble_parser = subparsers.add_parser("ensemble")
    ensemble_parser.add_argument("--dataset-id", required=True)
    ensemble_parser.add_argument("--models", nargs="+", default=["autoencoder", "vae", "transformer"])
    ensemble_parser.add_argument("--threshold-percentile", type=float, default=95.0)

    agent_parser = subparsers.add_parser("agent-run")
    agent_parser.add_argument("--dataset-id", required=True)
    agent_parser.add_argument("--models", nargs="+", default=["autoencoder", "vae", "transformer"])
    agent_parser.add_argument("--epochs", type=int, default=3)
    agent_parser.add_argument("--cpu", action="store_true")

    infra_parser = subparsers.add_parser("infra-status")
    infra_parser.add_argument("--live-probe", action="store_true")

    signing_status_parser = subparsers.add_parser("signing-status")

    rotate_signing_parser = subparsers.add_parser("rotate-signing")
    rotate_signing_parser.add_argument("--scope", default="all")
    rotate_signing_parser.add_argument("--index", type=int, default=None)

    args = parser.parse_args()
    token_file = Path(args.token_file)
    token = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else None
    sdk = AstroSDK(base_url=args.base_url, token=token)

    if args.command == "login":
        result = sdk.login(args.email, args.password)
        token_file.write_text(result["access_token"], encoding="utf-8")
        print(json.dumps({"status": "ok", "token_file": str(token_file)}, indent=2))
        return
    if args.command == "train":
        print(json.dumps(sdk.train(args.dataset_id, args.models, epochs=args.epochs), indent=2))
        return
    if args.command == "detect":
        print(
            json.dumps(
                sdk.detect(
                    args.dataset_id,
                    args.model_name,
                    threshold_percentile=args.threshold_percentile,
                ),
                indent=2,
            )
        )
        return
    if args.command == "agent-run":
        print(
            json.dumps(
                sdk.run_agent_cycle(
                    args.dataset_id,
                    args.models,
                    epochs=args.epochs,
                    use_gpu=not args.cpu,
                ),
                indent=2,
            )
        )
        return
    if args.command == "infra-status":
        print(json.dumps(sdk.infra_status(live_probe=args.live_probe), indent=2))
        return
    if args.command == "signing-status":
        print(json.dumps(sdk.signing_status(), indent=2))
        return
    if args.command == "rotate-signing":
        print(json.dumps(sdk.rotate_signing(scope=args.scope, index=args.index), indent=2))
        return
    print(
        json.dumps(
            sdk.ensemble_discover(
                args.dataset_id,
                args.models,
                threshold_percentile=args.threshold_percentile,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
