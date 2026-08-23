import json

from fastapi.testclient import TestClient

from trisynapse_memory.api import create_app


def test_api_exposes_health_and_empty_catalog(tmp_path):
    client = TestClient(create_app(tmp_path))

    health = client.get("/api/health")
    runs = client.get("/api/benchmarks/runs")
    catalog = client.get("/api/benchmarks/catalog")

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert runs.json() == {"runs": []}
    assert catalog.json() == {"schema_version": 1, "benchmarks": []}


def test_api_rejects_unsafe_run_paths(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/benchmarks/runs/locomo/%2E%2E")
    assert response.status_code in {400, 404}


def test_api_discovers_only_production_benchmark_artifacts(tmp_path):
    runs_root = tmp_path / "locomo" / "runs"
    runs_root.mkdir(parents=True)
    artifact = {
        "run_id": "20260804T010203Z",
        "benchmark": "locomo",
        "engine_version": "0.3.0",
        "architecture": "trace-and-recall-production",
        "summary": {"questions": 1},
        "store_validation": {"ok": True},
        "results": [],
    }
    (runs_root / "trace_recall_locomo_20260804T010203Z.json").write_text(json.dumps(artifact), encoding="utf-8")
    client = TestClient(create_app(tmp_path))

    runs = client.get("/api/v1/benchmarks/runs")
    catalog = client.get("/api/benchmarks/catalog")
    run = client.get("/api/v1/benchmarks/runs/locomo/20260804T010203Z")

    assert runs.status_code == 200 and runs.json()["runs"][0]["architecture"] == "trace-and-recall-production"
    assert catalog.json()["benchmarks"][0]["name"] == "locomo"
    assert run.status_code == 200 and run.json() == artifact


def test_api_rejects_removed_legacy_benchmark_contract(tmp_path):
    client = TestClient(create_app(tmp_path))
    removed_suite = client.post("/api/v1/benchmarks/runs", json={"benchmark": "longmemeval_v2"})
    removed_config = client.post(
        "/api/v1/benchmarks/runs",
        json={"benchmark": "locomo", "config_path": "configs/locomo.yaml"},
    )

    assert removed_suite.status_code == 422
    assert removed_config.status_code == 422
