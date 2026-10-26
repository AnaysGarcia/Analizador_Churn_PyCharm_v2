# Analizador y predictor de churn (PyCharm) — versión 2

Aplicación local en Python con interfaz web de Streamlit, preparada siguiendo el informe científico de Tokio Telecom adjunto.

## Funciones

- Carga de CSV con detección automática de coma, punto y coma o tabulación.
- Mapeo de cabeceras: ID de cliente, variable objetivo, valor que significa churn, tipo de tarifa y duración del contrato.
- Selección manual de variables categóricas y numéricas. El ID y el objetivo no se utilizan como predictores.
- Análisis del porcentaje de churn por tarifa y contrato.
- Estadísticas descriptivas, matriz de correlación y gráficos.
- Comparación de Random Forest, Regresión Logística y Gradient Boosting.
- Separación entrenamiento/test configurable (por defecto 80/20, semilla 42), validación cruzada de 3 folds y Grid Search.
- Métricas AUC-ROC, AUC-PR, Accuracy, Precision, Recall, especificidad, F1 y matriz de confusión.
- Curvas ROC, importancia agregada de variables de los modelos de árboles y prueba de robustez quitando la variable de contrato.
- Predicciones individuales, nivel de riesgo, alerta de retención y exportación CSV/Excel.
- Segundo CSV opcional para puntuar clientes nuevos, con mapeo de nombres diferente al histórico.
- Validación de archivos: límite de 20 MiB, 200.000 filas y 200 columnas; rechazo de cabeceras duplicadas y filas mal formadas.
- Exportaciones CSV/Excel con neutralización de texto que podría ejecutarse como fórmula en hojas de cálculo.
- Configuración local de Streamlit, estadísticas de uso del navegador desactivadas y mensajes de error prudentes.
- Código separado entre interfaz (`app.py`) y motor analítico (`churn_engine.py`), con docstrings y pruebas automatizadas.

## Abrir y ejecutar en PyCharm

1. Descarga y descomprime el ZIP del proyecto.
2. En PyCharm, abre la carpeta `Analizador_Churn_PyCharm`.
3. Usa Python 3.10 o 3.11 y crea un entorno virtual cuando PyCharm lo ofrezca.
4. Abre la terminal integrada de PyCharm y ejecuta:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Ejecuta la aplicación desde esa misma terminal:

   ```bash
   streamlit run app.py
   ```

6. Se abrirá una pestaña del navegador con la interfaz local. Si no se abre automáticamente, utiliza la URL que muestra Streamlit, habitualmente `http://localhost:8501`.

No ejecutes `app.py` con el botón verde de Python como si fuera un script normal; Streamlit debe iniciarlo con `streamlit run app.py`.

## Uso

1. Carga un **CSV histórico que incluya churn real** (por ejemplo, `Churn_YesNo` con valores `Yes`/`No`). Es imprescindible contar con etiquetas históricas para entrenar y evaluar un modelo supervisado.
2. Comprueba la vista previa y los tipos de datos detectados.
3. En el mapeo, selecciona la columna del ID (si existe), el objetivo, el valor positivo que representa churn, la tarifa y el contrato.
4. Revisa las selecciones de variables categóricas y numéricas. Las cabeceras originales pueden llamarse como quiera tu empresa.
5. Mantén la partición 80/20, semilla 42, 3 folds y Grid Search para seguir el enfoque del informe. Desmarca Grid Search si solo necesitas una prueba rápida.
6. Pulsa **Ejecutar análisis y entrenar modelos**. Navega por las pestañas y descarga el Excel o CSV.
7. Para predecir clientes que aún no tienen etiqueta, utiliza el cargador opcional de **CSV de clientes a puntuar** y mapea sus cabeceras con las utilizadas durante el entrenamiento.

## Modelo y evaluación

La aplicación implementa el enfoque metodológico del informe con scikit-learn en lugar de exigir una instalación local de Apache Spark. Mantiene los tres algoritmos indicados y compara su rendimiento con AUC-ROC. Para Random Forest y Gradient Boosting se usan los rangos estructurales del informe; la malla de Regresión Logística se adapta a los parámetros disponibles en scikit-learn. La búsqueda de hiperparámetros utiliza estos rangos:

- Random Forest: `max_depth` 5/10 y `n_estimators` 50/100.
- Regresión Logística: `C` 0.1/1/10 con el solver liblinear; es una adaptación de la malla de regularización (`regParam`/`elasticNetParam`) de PySpark a parámetros disponibles en scikit-learn.
- Gradient Boosting: `max_depth` 3/5 y `n_estimators` 50/100 (equivalente al `maxDepth`/`maxIter` descrito en el informe).

Las métricas publicadas en el informe (Accuracy de Random Forest 98,91 % y AUC-ROC 0,9972) son referencias del conjunto de datos usado en ese documento, **no resultados garantizados**. El programa calcula métricas nuevas para cada archivo, usando un test reservado que no participa en la optimización.

Tras evaluar los modelos, la aplicación vuelve a ajustar los mejores pipelines sobre todas las filas que tienen etiqueta para puntuar clientes. Por eso, las predicciones del CSV histórico no deben confundirse con las métricas independientes de test.

## Regla operativa de alertas

Si reconoce una columna equivalente a llamadas a soporte o retrasos en pagos, marca una alerta prioritaria cuando las llamadas superan 3 o el retraso de pago es superior a 0, en línea con la recomendación del informe. Los niveles de riesgo son umbrales iniciales de esta aplicación (alto ≥ 70 %, medio ≥ 40 %); ajústalos y valídalos con los responsables del negocio antes de automatizar campañas.

## CSV sintético de demostración

En `data/sample_customer_churn.csv` se incluye un conjunto **sintético y ficticio** para explorar la interfaz sin datos de clientes. No es el CSV original del informe y no debe usarse para evaluar un modelo real.

## Pruebas

La versión 2 incluye 8 pruebas automatizadas. Para ejecutarlas:

```bash
python -m unittest discover -s tests -v
```

## Seguridad y privacidad (versión 2)

Consulta `SECURITY.md` para el detalle de controles y una lista de comprobación previa al uso con datos reales. Entre las medidas incorporadas:

- Streamlit queda ligado a `127.0.0.1` por defecto; límite de subida configurado a 20 MiB.
- Validación de tamaño, estructura, filas, columnas y cabeceras duplicadas; mensajes de error sin volcar detalles de la librería.
- Protección contra inyección de fórmulas en las exportaciones CSV y Excel.
- Procesamiento local sin llamadas de red incorporadas para transmitir los CSV; estadísticas de uso del navegador desactivadas.

**Importante:** la aplicación no incorpora autenticación ni cifrado propio de archivos, y estas medidas no constituyen una certificación de cumplimiento del RGPD. No la expongas a Internet o a una red compartida sin controles de seguridad adicionales. Minimiza los datos personales y valida las predicciones antes de tomar decisiones comerciales. La importancia de una variable no demuestra causalidad.
