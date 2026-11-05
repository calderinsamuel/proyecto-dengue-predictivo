# Proyecto final: análisis de datos y aplicación predictiva de dengue pediátrico

Este proyecto se desarrolla en dos etapas, tal como se trabajará en Visual Studio Code:

1. **Cuaderno Jupyter:** exploración, limpieza, creación de variables, entrenamiento y evaluación de modelos.
2. **Página web:** aplicación Streamlit que carga los modelos guardados por el cuaderno y permite predicción individual y por lote.

La variable objetivo del Excel es `GROUPS`: 1 = dengue no complicado y 2 = dengue complicado, según el documento de referencia suministrado.

## 1. Dataset real integrado y artículo de referencia

Se incluye el artículo científico enviado por el profesor en `docs/articulo_referencia_pediatrico.pdf`. Su resumen, referencia bibliográfica y relación con los modelos del proyecto están documentados en `docs/ARTICULO_REFERENCIA.md`. El artículo original estudia 303 niños y desarrolla un puntaje con cinco indicadores mediante regresión logística; la red neuronal de este proyecto es una extensión académica, no un modelo presentado en ese artículo.


- Archivo: `data/MASTER_CHART_F1000Research.xlsx`.
- Hoja: `Sheet1`.
- Tamaño original: 303 registros y 25 columnas.
- Variable objetivo: primera columna, normalizada como `GROUPS`.
- Después de eliminar una fila completamente idéntica a otra: 302 registros, distribuidos en 239 de clase 1 y 63 de clase 2. Como no hay identificador de paciente, no puede confirmarse si la fila idéntica es un duplicado de digitación; se trata como supuesto de limpieza y el Excel original se conserva intacto.
- Valores faltantes: 112 en `FERRITIN`, uno en `ALT` y uno en `CREATININE`.

El documento de codificación describe 18 campos y menciona `OUTCOMES`, pero esa columna no aparece en el Excel. El proyecto no inventa ese campo y utiliza `GROUPS` como variable objetivo.

## 2. Parte A: cuaderno Jupyter en Visual Studio Code

Archivo principal: `notebooks/01_analisis_y_modelado_dengue.ipynb`.

El cuaderno realiza estos pasos:

1. Carga el Excel original y revisa dimensiones, tipos de datos, valores nulos y duplicados.
2. Normaliza los encabezados y el nombre de la variable objetivo.
3. Elimina duplicados, filas sin etiqueta, columnas vacías y predictores constantes.
4. Convierte los códigos categóricos documentados a etiquetas legibles (edad, género, síntomas y variables binarias).
5. Crea dos variables derivadas: `WBC_RANGE = HIGHEST WBC - LOWEST WBC` y `PLATELET_WBC_RATIO = PLATELET COUNT / HIGHEST WBC`.
6. Excluye por defecto `IV FLUIDS`, `ANTIBIOTICS`, `ANTIPYRETICS` y `STEROIDS` del entrenamiento debido al riesgo de fuga de información si se registraron después de conocer la gravedad o iniciar la atención.
7. Divide los datos con estratificación: 80 % entrenamiento y 20 % prueba, semilla aleatoria 42.
8. Entrena regresión logística y una red neuronal artificial MLP.
9. Evalúa accuracy, precisión, sensibilidad, F1, sensibilidad de clase 2, reportes de clasificación y matrices de confusión.
10. Guarda los modelos y resultados para que los utilice la página web.

### Archivos creados por el notebook

- `models/modelos_dengue.joblib`: pipelines entrenados, variables utilizadas, métricas y firma del dataset.
- `data/base_dengue_limpia.csv`: dataset después de limpieza y creación de variables.
- `data/metricas_modelos.json`: métricas y matrices de confusión.

## 3. Parte B: página web

Archivo principal: `app.py`.

La página utiliza `modeling.py`, el mismo módulo de limpieza y preparación que utiliza el cuaderno. Si encuentra el artefacto `models/modelos_dengue.joblib` y verifica que corresponde a la misma base, variable objetivo, variables predictoras y división 80/20, carga los modelos del cuaderno.

La aplicación incluye cuatro secciones:

- **Base y limpieza:** muestra el dataset procesado, resumen de limpieza, distribución de clases y descarga de CSV.
- **Entrenar y comparar:** muestra métricas y matrices de confusión cargadas desde el cuaderno; también permite volver a entrenar en la página.
- **Predicción individual:** recibe los valores de una observación y muestra predicción y probabilidades de ambos modelos.
- **Predicción por lote:** recibe un CSV o Excel, calcula predicciones para cada fila y permite descargar resultados.

En la pestaña **Entrenar y comparar** debe aparecer el mensaje “Modelos cargados desde el cuaderno Jupyter ejecutado en Visual Studio Code” cuando el artefacto corresponde exactamente a la base y configuración predeterminadas.

## 4. Métricas de referencia

Con 302 registros limpios, 22 variables predictoras y una partición estratificada 80/20 se obtuvieron 241 registros de entrenamiento y 61 de prueba.

| Métrica | Regresión logística | Red neuronal artificial |
|---|---:|---:|
| Accuracy | 0.9344 | 0.9016 |
| Precisión ponderada | 0.9334 | 0.9016 |
| Sensibilidad ponderada | 0.9344 | 0.9016 |
| F1 ponderado | 0.9324 | 0.9016 |
| Sensibilidad de clase 2 | 0.7692 | 0.7692 |

Estas métricas pertenecen a una sola partición interna del proyecto; no son los resultados publicados en el artículo científico. El artículo reportó para su puntaje de cinco indicadores sensibilidad de 84,13 %, especificidad de 72,50 % y AUC de 0,86. Las metodologías y variables no son idénticas, por lo que las cifras no constituyen una comparación directa.

Matrices de confusión (filas = clase real; columnas = clase predicha):

**Regresión logística**

| | Predicción 1 | Predicción 2 |
|---|---:|---:|
| Real 1 | 47 | 1 |
| Real 2 | 3 | 10 |

**Red neuronal**

| | Predicción 1 | Predicción 2 |
|---|---:|---:|
| Real 1 | 45 | 3 |
| Real 2 | 3 | 10 |

Son resultados de una sola división interna y de un conjunto pequeño y desbalanceado. No representan validación externa ni rendimiento clínico.

## 5. Cómo ejecutar en Visual Studio Code

1. Extrae el ZIP y abre la carpeta `proyecto_analitica_web` desde VS Code.
2. Instala las extensiones **Python** y **Jupyter** de Microsoft.
3. Instala Python 3.11 o 3.12.
4. Abre la terminal integrada de VS Code y ejecuta:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si tu equipo tiene Python 3.12, reemplaza `py -3.11` por `py -3.12`.

5. Abre `notebooks/01_analisis_y_modelado_dengue.ipynb`.
6. Pulsa **Select Kernel**, elige el intérprete Python de `.venv` y selecciona **Run All**.
7. Verifica que se creen `models/modelos_dengue.joblib`, `data/base_dengue_limpia.csv` y `data/metricas_modelos.json`.
8. Después ejecuta la página web desde la terminal:

```powershell
streamlit run app.py
```

También está disponible el archivo `INSTRUCCIONES_VISUAL_STUDIO_CODE.md` con los pasos ampliados.

## 6. Publicar la página web

Después de probarla localmente:

1. Crea un repositorio GitHub.
2. Sube el proyecto, incluyendo `app.py`, `modeling.py`, `requirements.txt`, `.streamlit/`, `data/` y `models/modelos_dengue.joblib`.
3. En Streamlit Community Cloud inicia sesión con GitHub, crea una app y selecciona el repositorio, la rama `main` y el archivo principal `app.py`.
4. Publica y copia la URL que genere el servicio.

Enlaces:
- https://github.com/new
- https://share.streamlit.io/
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## 7. Estructura del proyecto

```text
proyecto_analitica_web/
├── app.py
├── modeling.py
├── requirements.txt
├── README.md
├── MEMORIA_TECNICA.md
├── INSTRUCCIONES_VISUAL_STUDIO_CODE.md
├── launch_windows.bat
├── .streamlit/config.toml
├── notebooks/
│   └── 01_analisis_y_modelado_dengue.ipynb
├── models/
│   └── modelos_dengue.joblib
├── data/
│   ├── MASTER_CHART_F1000Research.xlsx
│   ├── dengue_pediatric_dataset.csv
│   ├── base_dengue_limpia.csv
│   ├── base_dengue_limpia.xlsx
│   └── metricas_modelos.json
├── docs/
│   ├── articulo_referencia_pediatrico.pdf
│   ├── ARTICULO_REFERENCIA.md
│   ├── FICHA_DATASET.md
│   └── README_codificacion_original.docx
└── tests/
    └── test_modeling.py
```

## Descarga de la base limpia

La aplicación ofrece dos formatos: **Excel (.xlsx)**, recomendado porque conserva las columnas y los tipos de datos, y **CSV con separador punto y coma**, pensado para abrirse en Excel con configuración regional española. El cuaderno Jupyter guarda ambos archivos en `data/`.

## 8. Limitaciones y uso responsable

- Las dos variables derivadas son transformaciones matemáticas exploratorias, no reglas clínicas ni umbrales diagnósticos.
- Se excluyen las variables de tratamiento por posible fuga de información. No se puede demostrar la temporalidad de todas las demás mediciones con los archivos disponibles.
- La clase complicada tiene menos registros que la clase no complicada; por ello se reporta la sensibilidad de clase 2 además del accuracy.
- No hay validación externa, datos longitudinales ni información suficiente para demostrar aplicabilidad en otros hospitales.
- Las predicciones son únicamente demostrativas y académicas. No deben utilizarse para diagnosticar, hacer triage ni decidir tratamientos.
