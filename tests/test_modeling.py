import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling import clean_dataset, train_models, prepare_dengue_prediction_data


def test_cleaning_removes_constant_and_keeps_target():
    raw = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "constant": ["same"] * 8,
        "target": ["yes", "no"] * 4,
    })
    clean, summary = clean_dataset(raw, "target")
    assert "constant" not in clean.columns
    assert "target" in clean.columns
    assert "constant" in summary["columnas_constantes_eliminadas"]


def test_cleaning_real_dengue_dataset_and_engineered_features():
    raw = pd.read_csv(ROOT / "data" / "dengue_pediatric_dataset.csv")
    clean, summary = clean_dataset(raw, "GROUPS")
    assert raw.shape == (303, 25)
    assert clean.shape == (302, 27)
    assert summary["filas_duplicadas_eliminadas"] == 1
    assert summary["clases"] == {"1": 239, "2": 63}
    assert summary["valores_faltantes_restantes"] == 114
    assert {"WBC_RANGE", "PLATELET_WBC_RATIO"}.issubset(clean.columns)
    assert clean["AGE"].isin(["Menor de 6 años", "Mayor de 6 años"]).all()
    assert clean["GENDER"].isin(["Masculino", "Femenino"]).all()
    assert clean["HEADACHE"].isin(["Sí", "No"]).all()


def test_batch_preparation_creates_same_dengue_features():
    raw = pd.read_csv(ROOT / "data" / "dengue_pediatric_dataset.csv").drop(columns=["GROUPS"]).head(5)
    prepared = prepare_dengue_prediction_data(raw)
    assert "WBC_RANGE" in prepared.columns
    assert "PLATELET_WBC_RATIO" in prepared.columns
    assert set(prepared["GENDER"].dropna().unique()).issubset({"Masculino", "Femenino"})


def test_models_train_on_real_dengue_data_without_treatment_predictors():
    raw = pd.read_csv(ROOT / "data" / "dengue_pediatric_dataset.csv")
    clean, _ = clean_dataset(raw, "GROUPS")
    excluded = {"IV FLUIDS", "ANTIBIOTICS", "ANTIPYRETICS", "STEROIDS"}
    features = [c for c in clean.columns if c != "GROUPS" and c not in excluded]
    result = train_models(clean, "GROUPS", features, test_size=0.2)
    assert set(result["models"].keys()) == {"Regresión logística", "Red neuronal artificial"}
    assert result["n_rows"] == 302
    assert result["n_train"] == 241
    assert result["n_test"] == 61
    assert result["evaluation"]["Regresión logística"]["confusion_matrix"].shape == (2, 2)
    assert 0 <= result["evaluation"]["Red neuronal artificial"]["accuracy"] <= 1


def test_notebook_saved_models_artifact_matches_dataset():
    artifact_path = ROOT / "models" / "modelos_dengue.joblib"
    assert artifact_path.exists()
    bundle = joblib.load(artifact_path)
    raw = pd.read_csv(ROOT / "data" / "dengue_pediatric_dataset.csv")
    clean, _ = clean_dataset(raw, "GROUPS")
    content = pd.util.hash_pandas_object(clean, index=True).values.tobytes()
    content += "|".join(map(str, clean.columns)).encode("utf-8", errors="ignore")
    import hashlib
    signature = hashlib.sha256(content).hexdigest()
    assert bundle["data_signature"] == signature
    assert bundle["target"] == "GROUPS"
    assert set(bundle["models"]) == {"Regresión logística", "Red neuronal artificial"}
