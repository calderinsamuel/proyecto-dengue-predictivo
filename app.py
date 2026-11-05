from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modeling import (
    clean_dataset,
    train_models,
)

APP_DIR = Path(__file__).parent
DEFAULT_DATASET_PATH = APP_DIR / "data" / "dengue_pediatric_dataset.csv"
NOTEBOOK_MODEL_PATH = APP_DIR / "models" / "modelos_dengue.joblib"

DENGUE_CLASS_LABELS = {"1": "Dengue no complicado", "2": "Dengue complicado"}

PREDICTOR_VALUE_LABELS = {
    "AGE": {1: "Menor de 6 años", 2: "Mayor de 6 años"},
    "GENDER": {1: "Masculino", 2: "Femenino"},
    "HEADACHE": {1: "Sí", 2: "No"},
    "MYALGIA": {1: "Sí", 2: "No"},
    "ABDOMINAL PAIN": {1: "Sí", 2: "No"},
    "RASH": {1: "Sí", 2: "No"},
    "VOMITING": {1: "Sí", 2: "No"},
    "BREATHLESSNESS": {1: "Sí", 2: "No"},
    "BLEEDING": {1: "Sí", 2: "No"},
    "BP AT ADMISSION": {1: "Sí", 2: "No"},
    "ORGANOMEGALY": {1: "Sí", 2: "No"},
    "IV FLUIDS": {1: "Sí", 2: "No"},
    "ANTIBIOTICS": {1: "Sí", 2: "No"},
    "ANTIPYRETICS": {1: "Sí", 2: "No"},
    "STEROIDS": {1: "Sí", 2: "No"},
}

TREATMENT_FEATURES = {"IV FLUIDS", "ANTIBIOTICS", "ANTIPYRETICS", "STEROIDS"}

st.set_page_config(
    page_title="Analítica predictiva | Proyecto final",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

/* Ocultar únicamente el menú principal */
#MainMenu {
    display: none !important;
}

/* Ocultar la decoración superior */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Ocultar el pie de página */
footer {
    display: none !important;
}

/* Mantener visible la cabecera y sus botones */
header[data-testid="stHeader"] {
    display: block !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""

<style>
    /* Fondo y colores adaptables al tema de Streamlit */
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    .main .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* Textos generales */
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp li,
    .stApp small,
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stCaption {
        color: var(--text-color);
    }

    /* Títulos */
    .stApp h1:not(.hero h1),
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: var(--text-color);
    }

    /* Encabezado principal */
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(115deg, #102a43, #176b87);
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        margin: 0 0 .35rem 0;
    }

    .hero p {
        color: #e2f2f7 !important;
        margin: 0;
        font-size: 1.02rem;
    }

    /* Tarjetas */
    .mini-card {
        border: 1px solid var(--secondary-background-color);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        background: var(--secondary-background-color);
        min-height: 100px;
    }

    .mini-card h3,
    .mini-card p {
        color: var(--text-color);
    }

    /* Métricas */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 12px 15px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }

    /* Pestañas */
    button[data-baseweb="tab"] {
        color: var(--text-color) !important;
    }

    button[data-baseweb="tab"] p {
        color: var(--text-color) !important;
    }

    /* Campos de entrada */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color);
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background: var(--secondary-background-color);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-color);
    }

    /* Tablas */
    .stDataFrame {
        color: var(--text-color);
    }

    /* Textos secundarios */
    .subtle {
        color: var(--text-color);
        opacity: 0.8;
        font-size: .92rem;
    }
</style>

""", unsafe_allow_html=True)


def read_uploaded_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            file.seek(0)
            return pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, sep=None, engine="python", encoding="latin-1")
    if name.endswith(".xlsx"):
        file.seek(0)
        return pd.read_excel(file, engine="openpyxl")
    raise ValueError("Formato no compatible. Sube un archivo CSV o XLSX.")


def standardize_dataset_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres del Excel de dengue para admitir el archivo original."""
    out = df.copy()
    names = []
    for i, raw_name in enumerate(out.columns):
        name = str(raw_name).strip() or f"columna_{i + 1}"
        upper = name.upper()
        if "GROUPS" in upper and "DENGUE" in upper:
            name = "GROUPS"
        names.append(name)
    seen = {}
    unique_names = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    out.columns = unique_names
    return out


def display_class_label(value, target: str = "GROUPS") -> str:
    label = str(value)
    return DENGUE_CLASS_LABELS.get(label, label) if target == "GROUPS" else label


def predictor_value_label(feature: str, value) -> str:
    mapping = PREDICTOR_VALUE_LABELS.get(feature, {})
    try:
        key = int(float(value))
        if float(value).is_integer() and key in mapping:
            return mapping[key]
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_signature(df: pd.DataFrame) -> str:
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    content += "|".join(map(str, df.columns)).encode("utf-8", errors="ignore")
    return hashlib.sha256(content).hexdigest()


def format_percent(value) -> str:
    return f"{float(value) * 100:.2f}%"


def matrix_dataframe(metrics: dict, target: str = "GROUPS") -> pd.DataFrame:
    labels = [display_class_label(x, target) for x in metrics["labels"]]
    return pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).rename_axis("Real / Predicción")


def complicated_recall(metrics: dict) -> float:
    labels = [str(x) for x in metrics["labels"]]
    if "2" not in labels:
        return float("nan")
    matrix = np.asarray(metrics["confusion_matrix"])
    i = labels.index("2")
    total = matrix[i, :].sum()
    return float(matrix[i, i] / total) if total else 0.0


def show_metrics(metrics: dict, prefix: str = "") -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud (accuracy)", format_percent(metrics["accuracy"]))
    c2.metric("Precisión ponderada", format_percent(metrics["precision"]))
    c3.metric("Sensibilidad ponderada", format_percent(metrics["recall"]))
    c4.metric("F1 ponderado", format_percent(metrics["f1"]))


def prediction_probability_table(model, row: pd.DataFrame, target: str = "GROUPS") -> tuple[str, pd.DataFrame]:
    prediction = display_class_label(model.predict(row)[0], target)
    output = pd.DataFrame({"Clase": [display_class_label(x, target) for x in model.classes_.astype(str)], "Probabilidad": model.predict_proba(row)[0]})
    output["Probabilidad"] = output["Probabilidad"].astype(float)
    output = output.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    output["Probabilidad (%)"] = (output["Probabilidad"] * 100).map(lambda x: f"{x:.2f}%")
    return prediction, output[["Clase", "Probabilidad (%)"]]


def create_input_row(features: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        s = reference[feature]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                distinct = sorted(pd.Series(s.dropna().unique()).tolist())
                all_integer = bool(distinct) and all(float(v).is_integer() for v in distinct)
                if 0 < len(distinct) <= 10 and all_integer:
                    options = [int(v) for v in distinct]
                    median = int(pd.to_numeric(s, errors="coerce").median())
                    try:
                        idx = options.index(median)
                    except ValueError:
                        idx = 0
                    values[feature] = st.selectbox(
                        feature, options, index=idx,
                        format_func=lambda value, f=feature: predictor_value_label(f, value),
                        key=f"individual_{feature}"
                    )
                else:
                    median = pd.to_numeric(s, errors="coerce").median()
                    default = float(median) if pd.notna(median) else 0.0
                    step = 1.0 if all_integer else 0.1
                    values[feature] = st.number_input(feature, value=default, step=step, key=f"individual_{feature}")
            else:
                clean_values = s.dropna()
                unique_values = sorted(clean_values.astype(str).unique().tolist())
                if not unique_values:
                    values[feature] = st.text_input(feature, key=f"individual_{feature}")
                elif len(unique_values) <= 30:
                    mode_series = clean_values.astype(str).mode()
                    default = str(mode_series.iloc[0]) if not mode_series.empty else unique_values[0]
                    idx = unique_values.index(default) if default in unique_values else 0
                    values[feature] = st.selectbox(feature, unique_values, index=idx, key=f"individual_{feature}")
                else:
                    mode_series = clean_values.astype(str).mode()
                    default = str(mode_series.iloc[0]) if not mode_series.empty else ""
                    values[feature] = st.text_input(feature, value=default, key=f"individual_{feature}")
    return pd.DataFrame([values], columns=features)


st.markdown("""
<div class="hero">
  <h1>Analítica predictiva</h1>
  <p>  Regresión logística y redes neuronales artificiales</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("1. Base de datos")
    uploaded = st.file_uploader("Sube otra base de datos (opcional)", type=["csv", "xlsx"], help="CSV o Excel (.xlsx). Si no seleccionas un archivo, se usa el dataset real de dengue incluido en el ZIP.")
    if uploaded is None:
        st.info("Dataset cargado: MASTER_CHART_F1000Research.xlsx (copia CSV normalizada incluida para lectura automática).")
    st.divider()

try:
    if uploaded is None:
        raw_df = pd.read_csv(DEFAULT_DATASET_PATH)
        source_name = "MASTER_CHART_F1000Research.xlsx (CSV normalizado)"
    else:
        raw_df = read_uploaded_file(uploaded)
        source_name = uploaded.name
except Exception as exc:
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

if raw_df.empty or raw_df.shape[1] < 2:
    st.error("La base debe contener al menos una fila y dos columnas: variables predictoras y una variable objetivo.")
    st.stop()

# Normalize headers, including the long target name from the original workbook.
raw_df = standardize_dataset_headers(raw_df)

st.caption(f"**Archivo:** {source_name} · **Dimensiones originales:** {raw_df.shape[0]:,} filas × {raw_df.shape[1]:,} columnas")

col_target, col_split = st.columns([2, 1])
with col_target:
    preferred_target = next((c for c in raw_df.columns if str(c).strip().upper() == "GROUPS"), None)
    if preferred_target is None:
        preferred_target = next((c for c in raw_df.columns if "GROUPS" in str(c).upper() and "DENGUE" in str(c).upper()), None)
    default_index = raw_df.columns.get_loc(preferred_target) if preferred_target else len(raw_df.columns) - 1
    target = st.selectbox(" Variable objetivo que deseas predecir", list(raw_df.columns), index=default_index)
with col_split:
    test_pct = st.slider("Porcentaje para prueba", min_value=20, max_value=40, value=20, step=5)

try:
    clean_df, clean_summary = clean_dataset(raw_df, target)
except Exception as exc:
    st.error(f"No se pudo limpiar la base: {exc}")
    st.stop()

features_available = [c for c in clean_df.columns if c != target]
with st.expander(" Selección de variables predictoras", expanded=False):
    recommended_features = [c for c in features_available if c not in TREATMENT_FEATURES]
    selected_features = st.multiselect(
        "Columnas que usará el modelo",
        options=features_available,
        default=recommended_features,
        help="Por defecto se excluyen tratamientos que pueden ocurrir después de conocer la gravedad, para reducir el riesgo de fuga de información.",
    )
    treatment_selected = sorted(set(selected_features) & TREATMENT_FEATURES)
    if treatment_selected:
        st.warning("Variables de tratamiento seleccionadas: " + ", ".join(treatment_selected) + ". Pueden introducir fuga de información si fueron registradas después de conocer la gravedad o iniciar la atención. Evalúa el momento de medición antes de interpretar el resultado.")

signature = dataframe_signature(clean_df)
current_key = (signature, target, tuple(selected_features), test_pct)
if st.session_state.get("current_key") != current_key:
    st.session_state.pop("trained_bundle", None)
    st.session_state.pop("bundle_origin", None)
    st.session_state["current_key"] = current_key

# Si el notebook ya entrenó los modelos con esta misma base y selección, la web los carga.
if "trained_bundle" not in st.session_state and NOTEBOOK_MODEL_PATH.exists():
    try:
        saved_bundle = joblib.load(NOTEBOOK_MODEL_PATH)
        compatible = (
            saved_bundle.get("target") == target
            and saved_bundle.get("features") == selected_features
            and saved_bundle.get("data_signature") == signature
            and abs(float(saved_bundle.get("test_size", 0)) - test_pct / 100) < 1e-9
        )
        if compatible:
            saved_bundle["reference_df"] = clean_df[selected_features].copy()
            saved_bundle["clean_summary"] = clean_summary
            saved_bundle["data_signature"] = signature
            st.session_state["trained_bundle"] = saved_bundle
            st.session_state["bundle_origin"] = "cuaderno Jupyter de Visual Studio Code"
    except Exception as exc:
        st.warning(f"No se pudieron cargar los modelos guardados desde el cuaderno: {exc}")

# Three functional views.
tab_data, tab_train, tab_individual = st.tabs([
    "Base y limpieza",
    "Entrenar y comparar",
    "Predicción individual"
])

with tab_data:
    st.subheader("Exploración y limpieza de datos")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filas después de limpiar", f"{len(clean_df):,}")
    k2.metric("Variables predictoras", len(features_available))
    k3.metric("Clases objetivo", int(clean_df[target].nunique()))
    k4.metric("Valores faltantes restantes", f"{clean_df[features_available].isna().sum().sum():,}")

    with st.expander("Registro de operaciones de limpieza", expanded=True):
        st.markdown(f"- Filas duplicadas eliminadas: **{clean_summary['filas_duplicadas_eliminadas']}**")
        st.markdown(f"- Filas sin valor objetivo eliminadas: **{clean_summary['filas_sin_objetivo_eliminadas']}**")
        st.markdown(f"- Columnas vacías eliminadas: **{', '.join(clean_summary['columnas_vacias_eliminadas']) if clean_summary['columnas_vacias_eliminadas'] else 'ninguna'}**")
        st.markdown(f"- Columnas constantes eliminadas: **{', '.join(clean_summary['columnas_constantes_eliminadas']) if clean_summary['columnas_constantes_eliminadas'] else 'ninguna'}**")
        st.markdown(f"- Columnas de texto convertidas a numéricas: **{', '.join(clean_summary['columnas_convertidas_a_numerico']) if clean_summary['columnas_convertidas_a_numerico'] else 'ninguna detectada'}**")
        st.markdown("- Valores faltantes en predictores: se imputan con la mediana (numéricos) o la moda (categóricos) durante el entrenamiento.")
        st.markdown("- Variables categóricas: se transforman a columnas numéricas mediante One-Hot Encoding.")
        st.markdown("- Variables numéricas: se estandarizan antes de entrenar los modelos.")
        st.markdown(f"- Variables derivadas creadas: **{', '.join(clean_summary.get('variables_derivadas_creadas', [])) or 'ninguna'}**.")
        st.markdown("- Códigos de edad, género y variables binarias: se convierten a etiquetas categóricas legibles antes de aplicar One-Hot Encoding.")

    st.markdown("**Vista previa de los datos limpios**")
    st.dataframe(clean_df.head(30), use_container_width=True, hide_index=True)

    # Exportación compatible con Excel: XLSX conserva las columnas y los tipos de datos.
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        clean_df.to_excel(writer, index=False, sheet_name="Base limpia")
        worksheet = writer.sheets["Base limpia"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        from openpyxl.styles import Font, PatternFill, Alignment
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="174A6E")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        worksheet.row_dimensions[1].height = 32
        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
            worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 35)
    excel_buffer.seek(0)

    # El CSV usa punto y coma como separador, habitual en Excel con configuración regional española.
    csv_clean = clean_df.to_csv(index=False, sep=";").encode("utf-8-sig")
    download_excel_col, download_csv_col = st.columns(2)
    with download_excel_col:
        st.download_button(
            "Descargar base limpia (Excel .xlsx)",
            data=excel_buffer.getvalue(),
            file_name="base_limpia.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with download_csv_col:
        st.download_button(
            "Descargar base limpia (CSV para Excel)",
            data=csv_clean,
            file_name="base_limpia.csv",
            mime="text/csv",
            help="Este CSV utiliza punto y coma como separador para que Excel distribuya los campos en columnas.",
            use_container_width=True,
        )
    st.markdown("**Distribución de la variable objetivo**")
    counts = clean_df[target].value_counts(dropna=False).rename_axis("Clase").reset_index(name="Registros")
    counts["Clase"] = counts["Clase"].map(lambda value: display_class_label(value, target))
    st.dataframe(counts, use_container_width=True, hide_index=True)

    if target == "GROUPS":
        st.info("Interpretación de GROUPS según el documento de referencia: 1 = dengue no complicado; 2 = dengue complicado.")

with tab_train:
    st.subheader("Entrenar los dos modelos")
    st.write("Ambos modelos usan la misma división de entrenamiento/prueba para que sus resultados sean comparables.")
    if not selected_features:
        st.warning("Selecciona al menos una variable predictora en la sección anterior.")
    if st.button("Entrenar regresión logística y red neuronal", type="primary", disabled=not bool(selected_features), use_container_width=True):
        with st.spinner("Limpiando y entrenando modelos..."):
            try:
                bundle = train_models(clean_df, target, selected_features, test_size=test_pct / 100)
                bundle["clean_summary"] = clean_summary
                bundle["data_signature"] = signature
                bundle["source_name"] = source_name
                bundle["reference_df"] = clean_df[selected_features].copy()
                st.session_state["trained_bundle"] = bundle
                st.session_state["bundle_origin"] = "entrenamiento realizado desde la página"
                st.success("Los dos modelos se entrenaron correctamente.")
            except Exception as exc:
                st.error(f"No fue posible entrenar los modelos: {exc}")

    bundle = st.session_state.get("trained_bundle")
    if bundle and bundle.get("data_signature") == signature and bundle.get("target") == target and bundle.get("features") == selected_features:
        st.markdown("### Resultados de prueba")
        if st.session_state.get("bundle_origin") == "cuaderno Jupyter de Visual Studio Code":
            st.success("Modelos cargados desde el cuaderno Jupyter ejecutado en Visual Studio Code.")
        st.caption(f"{bundle['n_rows']} registros evaluables · entrenamiento: {bundle['n_train']} · prueba: {bundle['n_test']} · división de prueba: {bundle['test_size']*100:.0f}%")
        model_names = list(bundle["evaluation"].keys())
        metric_rows = []
        for name in model_names:
            m = bundle["evaluation"][name]
            metric_rows.append({
                "Modelo": name,
                "Exactitud": m["accuracy"],
                "Precisión ponderada": m["precision"],
                "Sensibilidad ponderada": m["recall"],
                "F1 ponderado": m["f1"],
                ("Sensibilidad dengue complicado" if target == "GROUPS" else "Sensibilidad clase 2"): complicated_recall(m),
            })
        metrics_df = pd.DataFrame(metric_rows).set_index("Modelo")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        if target == "GROUPS":
            st.caption("La sensibilidad de dengue complicado se calcula sobre la clase 2 y ayuda a observar cuántos casos complicados detectó el modelo en el conjunto de prueba.")
        else:
            st.caption("La sensibilidad de la clase 2 se calcula sobre los registros cuya etiqueta real es 2.")

        for name in model_names:
            with st.expander(f"Matriz de confusión y reporte: {name}", expanded=(name == model_names[0])):
                m = bundle["evaluation"][name]
                st.dataframe(matrix_dataframe(m, target), use_container_width=True)
                st.caption("Filas = clase real; columnas = clase predicha.")
                st.code(m["report"], language="text")
                warning_message = bundle["convergence_warnings"].get(name)
                if warning_message:
                    st.warning(f"Aviso del entrenamiento: {warning_message}. La red terminó con una advertencia de convergencia; interpreta sus métricas con cautela.")

        best = max(bundle["evaluation"], key=lambda n: bundle["evaluation"][n]["f1"])
        st.info(f"En esta división de prueba, el mayor F1 ponderado fue de **{best}**. Esto no garantiza el mismo rendimiento con nuevos datos.")
        model_bytes = io.BytesIO()
        joblib.dump({"models": bundle["models"], "target": target, "features": selected_features, "version": 1}, model_bytes)
        st.download_button("Descargar modelos entrenados (.joblib)", data=model_bytes.getvalue(), file_name="modelos_analitica.joblib", mime="application/octet-stream")
    else:
        st.info("Pulsa **Entrenar** para calcular las métricas y habilitar las secciones de predicción.")

bundle = st.session_state.get("trained_bundle")
valid_bundle = bool(bundle and bundle.get("data_signature") == signature and bundle.get("target") == target and bundle.get("features") == selected_features)

with tab_individual:
    st.subheader("Predicción de un individuo")
    if not valid_bundle:
        st.info("Primero entrena los modelos en la pestaña **Entrenar y comparar**.")
    else:
        st.write("Introduce los valores de una observación. La aplicación mostrará la clase estimada por cada modelo y las probabilidades por clase.")
        with st.form("individual_form"):
            row = create_input_row(bundle["features"], bundle["reference_df"])
            submitted = st.form_submit_button("Predecir con ambos modelos", type="primary", use_container_width=True)
        if submitted:
            col1, col2 = st.columns(2)
            for column, model_name in zip([col1, col2], bundle["models"].keys()):
                with column:
                    prediction, probabilities = prediction_probability_table(bundle["models"][model_name], row, target)
                    st.markdown(f"#### {model_name}")
                    st.metric("Clase predicha", prediction)
                    st.dataframe(probabilities, use_container_width=True, hide_index=True)


