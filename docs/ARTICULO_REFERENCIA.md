# Artículo científico de referencia

## Identificación bibliográfica

Bhaskar, M., Mahalingam, S., M. M. H., & Achappa, B. (2022). *Predictive scoring system for risk of complications in pediatric dengue infection* (version 1). **F1000Research, 11**, 446. Publicado el 21 de abril de 2022. https://doi.org/10.12688/f1000research.111214.1

Archivo incluido: `articulo_referencia_pediatrico.pdf`.

## Qué estudió el artículo

El estudio fue una cohorte prospectiva hospitalaria de niños de 0 a 18 años con dengue confirmado. Incluyó 303 pacientes, clasificados en 240 casos de dengue no complicado y 63 de dengue complicado. El trabajo analizó variables demográficas, hallazgos clínicos y resultados de laboratorio.

Los autores desarrollaron un puntaje mediante regresión logística binaria y seleccionaron cinco indicadores:

| Indicador | Condición que suma puntos en el puntaje original |
|---|---|
| Hipotensión | Presente |
| Hematocrito / PCV | ≥ 42 % |
| Recuento de plaquetas | ≤ 75.000 células/mm³ |
| Glóbulos blancos / WBC más alto | ≥ 7.000 células/mm³ |
| ALT | ≥ 70 U/L |

El puntaje total iba de 0 a 7. Un umbral de 2 o más se asoció con dengue complicado. Para ese umbral, el artículo reportó sensibilidad de 84,13 %, especificidad de 72,50 % y AUC de 0,86 (IC 95 %: 0,80–0,92). Estos son resultados del artículo original, no de los modelos implementados en este proyecto.

## Relación con este proyecto académico

El artículo proporciona el contexto científico, la definición de la variable `GROUPS` y un referente metodológico para la clasificación. El proyecto usa el Excel suministrado como fuente real de datos y desarrolla dos modelos solicitados en la asignatura:

1. **Regresión logística**, entrenada con los predictores disponibles seleccionados en el cuaderno.
2. **Red neuronal artificial (MLP)**, como segundo enfoque de clasificación para comparar resultados.

La implementación no reproduce exactamente el puntaje clínico de cinco indicadores del artículo: utiliza 22 predictores disponibles en el conjunto de datos después de excluir cuatro variables de tratamiento por riesgo de fuga de información. El MLP es una extensión académica y no forma parte del artículo original.

En una única partición estratificada de prueba del proyecto (241 registros de entrenamiento y 61 de prueba), los resultados guardados son:

| Métrica | Regresión logística | Red neuronal artificial |
|---|---:|---:|
| Exactitud (accuracy) | 93,44 % | 90,16 % |
| Sensibilidad de la clase 2 | 76,92 % | 76,92 % |

No se deben comparar estas cifras como si fueran una validación directa entre estudios: cambian las variables, el método de evaluación y la partición. La muestra de prueba tiene solo 13 casos de clase 2 y no existe validación externa.

## Precauciones de interpretación

- El artículo señala que se necesita validación antes de aplicar el puntaje en la práctica clínica.
- Este proyecto es una demostración universitaria. No es un dispositivo médico y no debe usarse para diagnóstico, triage ni decisiones de tratamiento.
- El archivo Excel contiene 303 filas. El flujo de limpieza elimina una fila completamente idéntica a otra; como no hay identificador de paciente, no es posible verificar si era un duplicado de digitación o dos observaciones distintas con los mismos valores. Se conserva intacto el Excel original y esta decisión debe mencionarse como supuesto de limpieza.
- El documento de codificación enumera `OUTCOMES`, pero esa columna no está presente en el Excel entregado. No se crea ni se infiere esa variable.

## Licencia y atribución

El artículo se distribuye con licencia Creative Commons Attribution (CC BY), por lo que debe citarse la referencia bibliográfica anterior. El propio artículo indica que los datos subyacentes se publican bajo CC0 1.0. Para fines académicos, se conserva aquí el PDF de referencia entregado por el profesor y se atribuye explícitamente la publicación.
