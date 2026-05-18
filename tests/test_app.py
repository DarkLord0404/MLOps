import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, INTERPRETACIONES


def _make_client(tmp_path, monkeypatch):
    stats_file = tmp_path / "predicciones.jsonl"
    monkeypatch.setenv("PREDICTION_STATS_PATH", str(stats_file))
    app.config["TESTING"] = True
    return app.test_client(), stats_file


def test_prediccion_terminal_se_registra(tmp_path, monkeypatch):
    client, stats_file = _make_client(tmp_path, monkeypatch)

    payload = {
        "spo2": 82,
        "dolor": 9,
        "hemoglobina": 3.6,
        "fiebre": 39.8,
        "frecuencia_respiratoria": 42,
        "crisis_previas_6m": 6,
    }
    response = client.post("/predecir", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["resultado"] == "ENFERMEDAD TERMINAL"
    assert stats_file.exists()
    assert sum(1 for _ in stats_file.open("r", encoding="utf-8")) == 1


def test_reporte_estadisticas(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    response = client.get("/reporte")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ultimas_5"] == []
    assert data["fecha_ultima_prediccion"] is None
    for categoria in INTERPRETACIONES:
        assert data["total_por_categoria"][categoria] == 0

    payload = {
        "spo2": 96,
        "dolor": 5,
        "hemoglobina": 7.8,
        "fiebre": 38.6,
        "frecuencia_respiratoria": 20,
        "crisis_previas_6m": 1,
    }
    client.post("/predecir", json=payload)

    response = client.get("/reporte")
    data = response.get_json()
    assert data["total_por_categoria"]["ENFERMEDAD LEVE"] == 1
    assert data["ultimas_5"][-1]["resultado"] == "ENFERMEDAD LEVE"
