# Ficha técnica del dataset

## Identificación

**Título del estudio descrito en el Excel:** *Development of a Predictive Scoring System for Risk of Complications in Pediatric Dengue Infection*.

**Archivo:** `MASTER_CHART_F1000Research.xlsx`  
**Hoja:** `Sheet1`  
**Dimensión de origen:** 303 filas de datos × 25 columnas.

## Variable objetivo

| Campo | Código | Interpretación |
|---|---:|---|
| GROUPS | 1 | Dengue no complicado |
| GROUPS | 2 | Dengue complicado |

La variable `GROUPS` corresponde al primer campo del Excel, cuyo encabezado original es `GROUPS: DENGUE FEVER OR COMPLICATED DENGUE`.

## Variables de entrada y codificación

| Campo en Excel | Tipo / codificación descrita |
|---|---|
| AGE | 1 = menor de 6 años; 2 = mayor de 6 años |
| GENDER | 1 = masculino; 2 = femenino |
| DURATION OF FEVER | Número de días |
| HEADACHE | 1 = sí; 2 = no |
| MYALGIA | 1 = sí; 2 = no |
| ABDOMINAL PAIN | 1 = sí; 2 = no |
| RASH | 1 = sí; 2 = no |
| VOMITING | 1 = sí; 2 = no |
| BREATHLESSNESS | 1 = sí; 2 = no |
| BLEEDING | 1 = sí; 2 = no |
| BP AT ADMISSION | 1 = sí; 2 = no |
| ORGANOMEGALY | 1 = sí; 2 = no |
| HB | Valor numérico de laboratorio |
| PCV | Valor numérico de laboratorio |
| LOWEST WBC | Valor numérico de laboratorio |
| HIGHEST WBC | Valor numérico de laboratorio |
| PLATELET COUNT | Valor numérico de laboratorio |
| ALT | Valor numérico de laboratorio |
| CREATININE | Valor numérico de laboratorio |
| FERRITIN | Valor numérico de laboratorio |
| IV FLUIDS | 1 = sí; 2 = no |
| ANTIBIOTICS | 1 = sí; 2 = no |
| ANTIPYRETICS | 1 = sí; 2 = no |
| STEROIDS | 1 = sí; 2 = no |

## Calidad de datos y limpieza aplicada

| Indicador | Resultado |
|---|---:|
| Registros de origen | 303 |
| Columnas de origen | 25 |
| Filas completamente idénticas eliminadas | 1 |
| Registros sin etiqueta objetivo eliminados | 0 |
| Columnas completamente vacías eliminadas | 0 |
| Columnas predictoras constantes eliminadas | 0 |
| Registros finales | 302 |
| Predictores disponibles | 24 |
| Casos de clase 1 después de limpiar | 239 |
| Casos de clase 2 después de limpiar | 63 |
| Valores faltantes después de quitar duplicados | 114 |
| Faltantes en FERRITIN | 112 |
| Faltantes en ALT | 1 |
| Faltantes en CREATININE | 1 |

Los faltantes se imputan dentro del pipeline de cada modelo: mediana para variables numéricas y moda para categóricas. El ajuste de imputación se realiza con el subconjunto de entrenamiento, evitando estimar esos valores con los datos reservados para prueba.

**Supuesto sobre la fila idéntica:** se eliminó una fila con todos sus campos iguales a otra. El Excel no trae un identificador único de paciente, por lo que no se puede determinar si es una duplicación de digitación o si representa dos observaciones distintas con los mismos valores. Se conserva el Excel fuente sin modificaciones y se documenta que los modelos se entrenaron con 302 filas tras esta decisión.

## Diferencias frente al documento de referencia

El documento `README_codificacion_original.docx` contiene una lista de 18 campos e incluye `OUTCOMES` (1 = discharged; 2 = expired). Sin embargo, `MASTER_CHART_F1000Research.xlsx` tiene 25 columnas: incluye ocho campos de laboratorio (HB, PCV, LOWEST WBC, HIGHEST WBC, PLATELET COUNT, ALT, CREATININE y FERRITIN), además de IV FLUIDS, ANTIBIOTICS, ANTIPYRETICS y STEROIDS; no contiene `OUTCOMES`. La aplicación trabaja con las columnas que están realmente presentes y utiliza `GROUPS` como variable objetivo.

## Nota sobre los campos de tratamiento

Los campos de administración de líquidos y medicamentos pueden haberse registrado después de la valoración inicial. Si se utilizan para predecir complicaciones en un momento anterior a su administración, podrían producir fuga de información. La aplicación permite seleccionar/excluir predictores y advierte cuando alguno de estos campos se mantiene. El momento exacto en que se registró cada variable no está especificado en el archivo entregado.

**Uso:** conjunto de datos y modelos destinados exclusivamente a una actividad académica. No se ha establecido validación clínica externa.


## Relación con el artículo científico

El PDF del artículo se incluye en `articulo_referencia_pediatrico.pdf`. Bhaskar et al. (2022) analizaron 303 niños (240 no complicados y 63 complicados) y construyeron un puntaje de 0 a 7 con cinco indicadores: hipotensión, PCV ≥ 42 %, plaquetas ≤ 75.000, WBC ≥ 7.000 y ALT ≥ 70 U/L. El umbral ≥ 2 reportó sensibilidad de 84,13 %, especificidad de 72,50 % y AUC de 0,86. Consulte `ARTICULO_REFERENCIA.md` para la referencia bibliográfica y la distinción entre el estudio original y los modelos del proyecto.
