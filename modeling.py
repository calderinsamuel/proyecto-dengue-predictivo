"""Funciones de limpieza, entrenamiento y evaluación para la aplicación."""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression



def normalize_label(value: Any) -> str:
    """Normaliza etiquetas numéricas para que, por ejemplo, 1 y 1.0 coincidan."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass
    return text


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia encabezados y hace únicos los nombres de columnas."""
    out = df.copy()
    seen: dict[str, int] = {}
    names: list[str] = []
    for raw_name in out.columns:
        name = str(raw_name).strip() or "columna_sin_nombre"
        count = seen.get(name, 0) + 1
        seen[name] = count
        names.append(name if count == 1 else f"{name}_{count}")
    out.columns = names
    return out


def normalize_text_and_numbers(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Normaliza espacios, celdas vacías e identifica columnas numéricas guardadas como texto."""
    out = df.copy()
    converted: list[str] = []
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            s = s.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "None": np.nan})
            if col != target:
                non_null = s.notna()
                if int(non_null.sum()) > 0:
                    numeric = pd.to_numeric(s, errors="coerce")
                    ratio = numeric[non_null].notna().mean()
                    if ratio >= 0.95:
                        s = numeric
                        converted.append(col)
            out[col] = s
    # Unifica categorías como texto para evitar columnas mixtas (p. ej. números y palabras).
    for col in out.columns:
        if col != target and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: np.nan if pd.isna(value) else str(value).strip())
    return out, converted



DENGUE_CODE_MAPPINGS: dict[str, dict[str, str]] = {
    "AGE": {"1": "Menor de 6 años", "2": "Mayor de 6 años"},
    "GENDER": {"1": "Masculino", "2": "Femenino"},
}
for _binary_column in (
    "HEADACHE", "MYALGIA", "ABDOMINAL PAIN", "RASH", "VOMITING",
    "BREATHLESSNESS", "BLEEDING", "BP AT ADMISSION", "ORGANOMEGALY",
    "IV FLUIDS", "ANTIBIOTICS", "ANTIPYRETICS", "STEROIDS",
):
    DENGUE_CODE_MAPPINGS[_binary_column] = {"1": "Sí", "2": "No"}


def map_dengue_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte códigos categóricos documentados del dataset de dengue a etiquetas legibles."""
    out = df.copy()
    for column, mapping in DENGUE_CODE_MAPPINGS.items():
        if column not in out.columns:
            continue
        def _map_value(value):
            if pd.isna(value):
                return np.nan
            key = normalize_label(value)
            return mapping.get(key, str(value).strip())
        out[column] = out[column].map(_map_value)
    return out


def add_dengue_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Crea indicadores exploratorios a partir de columnas de laboratorio existentes."""
    out = df.copy()
    created: list[str] = []
    if {"HIGHEST WBC", "LOWEST WBC"}.issubset(out.columns):
        high = pd.to_numeric(out["HIGHEST WBC"], errors="coerce")
        low = pd.to_numeric(out["LOWEST WBC"], errors="coerce")
        out["WBC_RANGE"] = high - low
        created.append("WBC_RANGE")
    if {"PLATELET COUNT", "HIGHEST WBC"}.issubset(out.columns):
        platelets = pd.to_numeric(out["PLATELET COUNT"], errors="coerce")
        high = pd.to_numeric(out["HIGHEST WBC"], errors="coerce").replace(0, np.nan)
        out["PLATELET_WBC_RATIO"] = platelets / high
        created.append("PLATELET_WBC_RATIO")
    return out, created


def prepare_dengue_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara nuevos registros con las mismas reglas categóricas y variables derivadas."""
    out = normalize_column_names(df)
    out, _ = normalize_text_and_numbers(out, target="__NO_TARGET__")
    out = map_dengue_codes(out)
    out, _ = add_dengue_derived_features(out)
    return out

def clean_dataset(raw: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica limpieza reproducible y devuelve el conjunto limpio y un resumen."""
    df = normalize_column_names(raw)
    if target not in df.columns:
        raise ValueError("La variable objetivo ya no aparece en los encabezados de la base.")

    rows_initial, cols_initial = df.shape
    df, converted = normalize_text_and_numbers(df, target)

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    missing_target = int(df[target].isna().sum())
    df = df.loc[df[target].notna()].copy()
    if df.empty:
        raise ValueError("No hay filas con una etiqueta válida en la variable objetivo.")

    empty_cols = [c for c in df.columns if c != target and df[c].isna().all()]
    df = df.drop(columns=empty_cols, errors="ignore")

    constant_cols = [
        c for c in df.columns
        if c != target and df[c].nunique(dropna=True) <= 1
    ]
    df = df.drop(columns=constant_cols, errors="ignore")

    # Las etiquetas se manejan como clases categóricas, incluso cuando llegan como números.
    df[target] = df[target].map(normalize_label)

    # Decodifica variables categóricas documentadas y crea dos indicadores exploratorios.
    df = map_dengue_codes(df)
    df, derived_features = add_dengue_derived_features(df)

    # Tras la creación de variables se vuelven a retirar predictores vacíos/constantes.
    empty_derived_cols = [c for c in df.columns if c != target and df[c].isna().all()]
    df = df.drop(columns=empty_derived_cols, errors="ignore")
    constant_cols_after_features = [
        c for c in df.columns
        if c != target and df[c].nunique(dropna=True) <= 1
    ]
    df = df.drop(columns=constant_cols_after_features, errors="ignore")
    constant_cols = list(dict.fromkeys(constant_cols + constant_cols_after_features))
    empty_cols = list(dict.fromkeys(empty_cols + empty_derived_cols))

    if df[target].nunique(dropna=False) < 2:
        raise ValueError("La variable objetivo debe tener al menos dos clases diferentes.")

    features = [c for c in df.columns if c != target]
    if not features:
        raise ValueError("No quedan variables predictoras después de quitar columnas vacías o constantes.")

    summary = {
        "filas_iniciales": rows_initial,
        "columnas_iniciales": cols_initial,
        "filas_duplicadas_eliminadas": duplicates_removed,
        "filas_sin_objetivo_eliminadas": missing_target,
        "columnas_vacias_eliminadas": empty_cols,
        "columnas_constantes_eliminadas": constant_cols,
        "columnas_convertidas_a_numerico": converted,
        "variables_derivadas_creadas": derived_features,
        "categorias_decodificadas": [c for c in DENGUE_CODE_MAPPINGS if c in df.columns],
        "filas_finales": int(df.shape[0]),
        "columnas_finales": int(df.shape[1]),
        "valores_faltantes_restantes": int(df[features].isna().sum().sum()),
        "clases": df[target].value_counts().to_dict(),
    }
    return df.reset_index(drop=True), summary


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputa faltantes, escala columnas numéricas y codifica categorías con One-Hot Encoding."""
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numericas", numeric_pipeline, numeric_cols),
            ("categoricas", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, labels: list[str] | np.ndarray | None = None) -> dict[str, Any]:
    """Calcula métricas para clasificación binaria o multiclase."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [str(label) for label in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def train_models(
    clean_df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Entrena regresión logística y red neuronal sobre la misma partición estratificada."""
    if not features:
        raise ValueError("Selecciona al menos una variable predictora.")
    data = clean_df[features + [target]].copy()
    data = data.loc[data[target].notna()].copy()
    X = data[features]
    y = data[target].astype(str)

    class_counts = y.value_counts()
    n_classes = int(y.nunique())
    if n_classes < 2:
        raise ValueError("La variable objetivo tiene menos de dos clases.")
    if int(class_counts.min()) < 2:
        rare = class_counts[class_counts < 2].index.tolist()
        raise ValueError(
            "Cada clase debe tener al menos 2 registros para separar entrenamiento y prueba. "
            f"Clases con un solo registro: {rare}."
        )
    if len(data) < max(8, n_classes * 3):
        raise ValueError("La base es demasiado pequeña para una evaluación de prueba confiable.")

    n_test = max(math.ceil(len(data) * test_size), n_classes)
    if len(data) - n_test < n_classes:
        raise ValueError("No hay suficientes filas para separar entrenamiento y prueba con todas las clases.")
    adjusted_test_size = n_test / len(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=adjusted_test_size,
        random_state=random_state,
        stratify=y,
    )

    estimators = {
        "Regresión logística": LogisticRegression(max_iter=2000, random_state=random_state),
        "Red neuronal artificial": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=700,
            early_stopping=False,
            random_state=random_state,
        ),
    }

    models: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    convergence_warnings: dict[str, str | None] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocesamiento", make_preprocessor(X_train)),
                ("modelo", estimator),
            ]
        )
        warning_message = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    warning_message = str(warning.message)
                    break
        prediction = pipeline.predict(X_test).astype(str)
        models[name] = pipeline
        metrics = evaluate_predictions(y_test, prediction, labels=sorted(y.unique().tolist()))
        evaluation[name] = metrics
        convergence_warnings[name] = warning_message

    return {
        "models": models,
        "evaluation": evaluation,
        "features": list(features),
        "target": target,
        "n_rows": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": float(adjusted_test_size),
        "class_counts": class_counts.to_dict(),
        "X_reference": X.reset_index(drop=True),
        "y_reference": y.reset_index(drop=True),
        "convergence_warnings": convergence_warnings,
    }
