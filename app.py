from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# La aplicación está pensada para ejecutarse localmente. No registres filas ni
# valores de clientes en logs o mensajes de error. Los CSV se procesan en memoria.

from churn_engine import (
    MODEL_GB,
    MODEL_LR,
    MODEL_ORDER,
    MODEL_RF,
    AnalysisConfig,
    coerce_numeric_series,
    find_column,
    frame_to_csv_bytes,
    infer_feature_columns,
    make_excel_report,
    predict_new_customers,
    read_csv_file,
    run_analysis,
    suggest_column_mappings,
)


st.set_page_config(page_title="Predicción de Churn | Tokio Telecom", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1450px;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 14px; background: linear-gradient(115deg,#10283f,#1b5270); color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .3rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .9;}
    .small-note {color: #687787; font-size: .88rem;}
    </style>
    <div class="hero">
      <h1>Analizador y predictor de churn</h1>
      <p>Tokio Telecom · análisis descriptivo, comparación de modelos y priorización de clientes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DEMO_PATH = BASE_DIR / "data" / "sample_customer_churn.csv"


def _file_fingerprint(name: str, content: bytes) -> str:
    return hashlib.sha256(name.encode("utf-8") + content).hexdigest()[:16]


def _default_option(options: list[str], candidate: str | None, none_label: str = "(No usar)") -> str:
    return candidate if candidate in options else none_label


def _fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.2f}%"


def _model_metrics_for_cards(metrics: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = metrics[(metrics["Modelo"] == model_name) & (metrics["Conjunto"] == "Test")]
    return None if rows.empty else rows.iloc[0]


def _display_segment_table(table: pd.DataFrame, segment_label: str) -> None:
    if table.empty:
        st.info(f"Selecciona la columna de {segment_label.lower()} en el mapeo para ver este análisis.")
        return
    st.dataframe(table, use_container_width=True, hide_index=True)


def _plot_segment(table: pd.DataFrame, title: str) -> None:
    if table.empty:
        return
    plotted = table.copy()
    plotted["Segmento"] = plotted["Segmento"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plotted, x="Segmento", y="Churn (%)", hue="Segmento", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Churn (%)")
    ax.set_ylim(0, min(100, max(10, float(plotted["Churn (%)"].max()) * 1.15)))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.sidebar:
    st.header("1. Cargar datos")
    uploaded_file = st.file_uploader("CSV histórico con una columna de churn conocida", type=["csv"], key="historical_csv")
    use_demo = st.checkbox("Usar CSV sintético de demostración", value=False, help="Son datos inventados para comprobar la aplicación; no son clientes reales ni los datos del informe.")
    if use_demo and uploaded_file is not None:
        st.warning("Se usará el CSV de demostración. Quita la selección para analizar el archivo subido.")
    st.caption("El separador (coma, punto y coma o tabulación) se detecta automáticamente. Límite de carga: 20 MiB; máximo 200.000 filas y 200 columnas.")

source_name: str | None = None
source_bytes: bytes | None = None
source_fingerprint: str | None = None
raw_df: pd.DataFrame | None = None

try:
    if use_demo:
        source_bytes = DEMO_PATH.read_bytes()
        source_name = DEMO_PATH.name
        raw_df = read_csv_file(io.BytesIO(source_bytes))
    elif uploaded_file is not None:
        source_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
        raw_df = read_csv_file(io.BytesIO(source_bytes))
except Exception:
    st.error("No se pudo leer el archivo. Comprueba el tamaño, la codificación, las cabeceras y el formato del CSV.")

if raw_df is None:
    st.info("Carga un CSV histórico para empezar. El archivo debe incluir una columna con el resultado real del churn (por ejemplo, Yes/No), porque esa columna se utiliza para entrenar y evaluar los modelos.")
    st.markdown("**Columnas habituales del informe:** `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage_Frequency`, `Support_Calls`, `Payment_Delay`, `Subscription_Type`, `Contract_Length`, `Total_Spend`, `Last_Interaction` y `Churn_YesNo`.")
    st.stop()

source_fingerprint = _file_fingerprint(source_name or "datos.csv", source_bytes or b"")
columns = list(raw_df.columns)
mapping_suggestions = suggest_column_mappings(columns)
def_cat, def_num = infer_feature_columns(raw_df, mapping_suggestions["target"], mapping_suggestions["id"])
no_use = "(No usar)"
selectable = [no_use] + columns

st.subheader("Vista previa del CSV")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Filas", f"{len(raw_df):,}")
col_b.metric("Columnas", f"{len(raw_df.columns):,}")
col_c.metric("Tamaño", f"{len(source_bytes or b'') / 1024:.1f} KB")
with st.expander("Ver primeras filas y tipos de datos", expanded=False):
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
    schema = pd.DataFrame({"Cabecera CSV": raw_df.columns, "Tipo detectado": [str(t) for t in raw_df.dtypes], "Valores no vacíos": [int(raw_df[c].notna().sum()) for c in raw_df.columns], "Valores distintos": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns]})
    st.dataframe(schema, use_container_width=True, hide_index=True)

st.subheader("2. Mapear las cabeceras del CSV")
st.caption("Selecciona qué columna significa cada cosa para tu negocio. Los nombres originales del archivo no se modifican.")
with st.container(border=True):
    col_id, col_target, col_positive = st.columns(3)
    id_default = _default_option(selectable, mapping_suggestions["id"])
    target_default = _default_option(selectable, mapping_suggestions["target"])
    id_choice = col_id.selectbox("Identificador del cliente (se excluye del modelo)", selectable, index=selectable.index(id_default), key="map_id")
    target_choice = col_target.selectbox("Columna que contiene el churn real", selectable, index=selectable.index(target_default), key="map_target")
    target_actual = None if target_choice == no_use else target_choice
    target_values: list[str] = []
    if target_actual:
        target_values = sorted(raw_df[target_actual].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].unique().tolist())
    normalized_defaults = {v.lower().strip(): v for v in target_values}
    positive_default = next((normalized_defaults[v] for v in ["yes", "si", "sí", "1", "true", "churn", "abandona", "baja"] if v in normalized_defaults), target_values[-1] if target_values else "")
    if target_values:
        positive_idx = target_values.index(positive_default) if positive_default in target_values else 0
        positive_value = col_positive.selectbox("Valor que significa abandono (clase positiva)", target_values, index=positive_idx, key=f"positive_{target_actual}")
        if len(target_values) > 2:
            st.warning("La columna objetivo tiene más de dos valores. El valor seleccionado se tratará como churn (Yes) y todos los demás se agruparán como no churn (No).")
    else:
        positive_value = "Yes"
        col_positive.selectbox("Valor que significa abandono", ["Carga primero la columna objetivo"], disabled=True)

    col_sub, col_contract = st.columns(2)
    sub_default = _default_option(selectable, mapping_suggestions["subscription"])
    contract_default = _default_option(selectable, mapping_suggestions["contract"])
    sub_choice = col_sub.selectbox("Columna de tipo de tarifa (opcional)", selectable, index=selectable.index(sub_default), key="map_subscription")
    contract_choice = col_contract.selectbox("Columna de duración del contrato (opcional)", selectable, index=selectable.index(contract_default), key="map_contract")

    feature_candidates = [c for c in columns if c not in {target_actual, None if id_choice == no_use else id_choice}]
    default_cat = [c for c in def_cat if c in feature_candidates]
    default_num = [c for c in def_num if c in feature_candidates]
    st.markdown("**Variables predictoras**")
    st.caption("Por defecto se infieren los tipos. Ajusta las selecciones si, por ejemplo, un número representa una categoría o un número llegó como texto.")
    all_feat_col1, all_feat_col2 = st.columns(2)
    categorical_cols = all_feat_col1.multiselect("Tratar como categóricas", feature_candidates, default=default_cat, key="features_cat")
    numeric_default_filtered = [c for c in default_num if c not in categorical_cols]
    numeric_cols = all_feat_col2.multiselect("Tratar como numéricas", feature_candidates, default=numeric_default_filtered, key="features_num")
    overlap = sorted(set(categorical_cols) & set(numeric_cols))
    if overlap:
        st.warning("Estas columnas se han seleccionado como categóricas y numéricas a la vez: " + ", ".join(overlap) + ". Se tratarán como categóricas.")
    feature_cols_preview = list(dict.fromkeys(categorical_cols + [c for c in numeric_cols if c not in categorical_cols]))
    if target_actual in feature_cols_preview:
        st.error("La columna de churn no puede ser una variable predictora; quítala de las selecciones.")
    if id_choice != no_use and id_choice in feature_cols_preview:
        st.warning("El identificador del cliente se excluirá del modelo automáticamente.")

st.subheader("3. Configuración del entrenamiento")
config_col1, config_col2, config_col3 = st.columns(3)
test_size_pct = config_col1.slider("Datos reservados para prueba", min_value=10, max_value=40, value=20, step=5, help="El informe usa 20 % de test y 80 % de train.")
cv_folds = config_col2.selectbox("Pliegues de validación cruzada", [3, 5], index=0, help="El informe usa 3 folds.")
optimize = config_col3.checkbox("Optimizar hiperparámetros (Grid Search)", value=True, help="Explora combinaciones de parámetros usando AUC-ROC; puede tardar más con archivos grandes.")
if optimize:
    st.caption("La búsqueda de parámetros se ejecuta sobre el conjunto de entrenamiento. El conjunto de prueba queda separado para evaluar el rendimiento final.")

current_config_signature = json.dumps({
    "file": source_fingerprint,
    "id": id_choice,
    "target": target_actual,
    "positive": positive_value,
    "subscription": None if sub_choice == no_use else sub_choice,
    "contract": None if contract_choice == no_use else contract_choice,
    "categorical": categorical_cols,
    "numeric": numeric_cols,
    "test": test_size_pct,
    "folds": cv_folds,
    "optimize": optimize,
}, ensure_ascii=False, sort_keys=True)

run_col, _ = st.columns([1, 3])
run_clicked = run_col.button("Ejecutar análisis y entrenar modelos", type="primary", use_container_width=True, disabled=(target_actual is None or not feature_cols_preview))
if run_clicked:
    config = AnalysisConfig(
        target_col=target_actual,
        positive_value=positive_value,
        id_col=None if id_choice == no_use else id_choice,
        subscription_col=None if sub_choice == no_use else sub_choice,
        contract_col=None if contract_choice == no_use else contract_choice,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        test_size=test_size_pct / 100,
        random_state=42,
        cv_folds=int(cv_folds),
        optimize_hyperparameters=bool(optimize),
    )
    progress = st.progress(0, text="Preparando datos...")
    status = st.empty()
    def update_progress(current: int, total: int, model_name: str) -> None:
        progress.progress(int(current / total * 100), text=f"Entrenando {model_name} ({current}/{total})...")
        status.write(f"Modelo procesado: {model_name}")
    try:
        with st.spinner("El análisis está en marcha. La búsqueda de hiperparámetros puede tardar unos minutos."):
            result = run_analysis(raw_df, config, progress_callback=update_progress)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_signature"] = current_config_signature
        st.session_state.pop("new_predictions", None)
        progress.progress(100, text="Análisis completado")
        status.empty()
        st.success("Análisis completado correctamente.")
    except Exception:
        progress.empty()
        status.empty()
        st.error("No se pudo completar el análisis. Revisa la columna objetivo, las clases disponibles y las variables seleccionadas; si persiste, consulta la guía de solución de problemas.")

result = st.session_state.get("analysis_result")
result_is_current = result is not None and st.session_state.get("analysis_signature") == current_config_signature
if result is not None and not result_is_current:
    st.info("Has cambiado el archivo o la configuración de columnas. Pulsa **Ejecutar análisis y entrenar modelos** para actualizar los resultados.")

if result_is_current:
    st.divider()
    st.header("Resultados del análisis")
    best_name = result["best_model_name"]
    best_test = _model_metrics_for_cards(result["metrics"], best_name)
    churn_rate = result["summary"].get("Tasa churn real (%)", np.nan)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes con churn conocido", f"{result['summary']['Filas con churn conocido']:,}")
    k2.metric("Churn observado", f"{churn_rate:.2f}%" if pd.notna(churn_rate) else "—")
    k3.metric("Modelo recomendado por validación", best_name)
    k4.metric("AUC-ROC en test", f"{best_test['AUC-ROC']:.4f}" if best_test is not None and pd.notna(best_test["AUC-ROC"]) else "—")

    tab_overview, tab_segments, tab_models, tab_features, tab_predictions, tab_export = st.tabs([
        "Resumen ejecutivo", "Segmentación", "Modelos y métricas", "Variables influyentes", "Predicciones", "Descargas"
    ])
    with tab_overview:
        st.markdown("#### Resumen para negocio")
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(pd.DataFrame([{"Concepto": k, "Resultado": v} for k, v in result["summary"].items()]), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Lectura recomendada**")
            st.write(f"El modelo seleccionado es **{best_name}**, por haber obtenido el mayor AUC medio en validación cruzada. La fila `Test` de la tabla de métricas es la referencia para estimar el rendimiento sobre datos reservados.")
            st.write("Las predicciones con riesgo alto o alertas operativas pueden ayudar a priorizar acciones de retención, pero deben revisarse con criterios comerciales y de privacidad antes de contactar con un cliente.")
            st.warning("El informe original describe un 100 % de churn en contratos Monthly y un Accuracy de Random Forest del 98,91 %. Son hallazgos del dataset del informe, no condiciones impuestas a los datos que cargues.")
        st.markdown("#### Hallazgos automáticos del archivo cargado")
        contract_table = result["by_contract"]
        subscription_table = result["by_subscription"]
        if not contract_table.empty:
            highest_contract = contract_table.sort_values("Churn (%)", ascending=False).iloc[0]
            st.write(f"La tasa de churn más alta por duración contractual está en **{highest_contract['Segmento']}** ({highest_contract['Churn (%)']:.2f} %, {int(highest_contract['Churn'])} de {int(highest_contract['Clientes'])} clientes etiquetados).")
            monthly_rows = contract_table[contract_table["Segmento"].astype(str).str.contains(r"monthly|mensual", case=False, regex=True)]
            if not monthly_rows.empty and float(monthly_rows.iloc[0]["Churn (%)"]) >= 99.9:
                st.warning("El CSV cargado muestra un churn prácticamente del 100 % en contratos mensuales. Como señala el informe, es una anomalía que conviene auditar (alta de clientes, periodo de prueba, facturación o calidad de los datos); por sí sola no prueba la causa.")
        if not subscription_table.empty:
            highest_subscription = subscription_table.sort_values("Churn (%)", ascending=False).iloc[0]
            if subscription_table["Churn (%)"].min() >= 55 and (subscription_table["Churn (%)"].max() - subscription_table["Churn (%)"].min()) <= 15:
                st.write("Las tarifas presentan una tasa de abandono elevada y relativamente parecida. Esto sugiere estudiar factores operativos y contractuales además del precio.")
            else:
                st.write(f"Por tarifa, el churn más alto se observa en **{highest_subscription['Segmento']}** ({highest_subscription['Churn (%)']:.2f} %). Valida el tamaño de cada segmento antes de priorizar campañas.")
        if not result["importance"].empty and MODEL_RF in result["importance"].columns:
            top_vars = result["importance"].sort_values(MODEL_RF, ascending=False).head(4)["Variable"].astype(str).tolist()
            st.write("Variables con mayor importancia en Random Forest: **" + ", ".join(top_vars) + "**. La importancia mide contribución predictiva, no causalidad.")
        st.markdown("#### Estadísticas descriptivas de las variables numéricas")
        if result["descriptive"].empty:
            st.info("No hay variables numéricas seleccionadas para mostrar estadísticas.")
        else:
            st.dataframe(result["descriptive"], use_container_width=True, hide_index=True)
        st.markdown("#### Matriz de correlación")
        if result["correlation"].empty:
            st.info("Selecciona al menos dos variables numéricas para calcular una matriz de correlación.")
        else:
            fig, ax = plt.subplots(figsize=(max(6, len(result["correlation"]) * 0.8), max(4.5, len(result["correlation"]) * 0.65)))
            sns.heatmap(result["correlation"], annot=True, fmt=".2f", center=0, cmap="coolwarm", square=True, ax=ax)
            ax.set_title("Correlación entre variables numéricas")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab_segments:
        seg_left, seg_right = st.columns(2)
        with seg_left:
            st.markdown("#### Churn por tipo de tarifa")
            _display_segment_table(result["by_subscription"], "tipo de tarifa")
            _plot_segment(result["by_subscription"], "Tasa de churn por tarifa")
        with seg_right:
            st.markdown("#### Churn por duración del contrato")
            _display_segment_table(result["by_contract"], "duración del contrato")
            _plot_segment(result["by_contract"], "Tasa de churn por contrato")
        st.caption("La tasa de cada segmento es el porcentaje de filas etiquetadas como churn positivo sobre los clientes del segmento.")

    with tab_models:
        st.markdown("#### Comparativa de rendimiento")
        metric_cols = ["Modelo", "Conjunto", "CV AUC medio", "AUC-ROC", "AUC-PR", "Accuracy", "Precision (churn)", "Recall (churn)", "Especificidad", "F1 (churn)", "TN", "FP", "FN", "TP"]
        metrics_to_show = result["metrics"][metric_cols].copy()
        st.dataframe(metrics_to_show, use_container_width=True, hide_index=True)
        st.caption("Precision, Recall, F1 y la matriz de confusión toman Churn como clase positiva. AUC-ROC y Accuracy se muestran en train y test.")
        st.markdown("#### Curvas ROC en el conjunto de prueba")
        fig, ax = plt.subplots(figsize=(7, 5))
        for model_name, curve in result["roc_curves"].items():
            ax.plot(curve["fpr"], curve["tpr"], linewidth=2, label=f"{model_name} (AUC = {curve['auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC = 0.5)")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curvas ROC · test")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        confusion_model = st.selectbox("Matriz de confusión para el modelo", MODEL_ORDER, index=MODEL_ORDER.index(best_name), key="cm_model")
        row_test = result["metrics"][(result["metrics"]["Modelo"] == confusion_model) & (result["metrics"]["Conjunto"] == "Test")].iloc[0]
        cm = np.array([[row_test["TN"], row_test["FP"]], [row_test["FN"], row_test["TP"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predice No", "Predice Yes"], yticklabels=["Real No", "Real Yes"], cbar=False, ax=ax)
        ax.set_title(f"Matriz de confusión · {confusion_model}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("#### Hiperparámetros seleccionados")
        params_rows = [{"Modelo": name, "AUC medio CV": result["cv_scores"][name], "Parámetros": str(result["best_params"].get(name, {}))} for name in MODEL_ORDER]
        st.dataframe(pd.DataFrame(params_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Prueba de robustez sin la duración del contrato")
        robustness = result.get("robustness", pd.DataFrame())
        if robustness.empty:
            st.info("Esta prueba se ejecuta si la columna de duración del contrato también se selecciona como variable predictora y queda al menos otra variable disponible.")
        else:
            st.dataframe(robustness.round(4), use_container_width=True, hide_index=True)
            st.caption("La caída de AUC compara el modelo completo con otro entrenado sin Contract_Length, usando los mismos datos de train y test y los hiperparámetros obtenidos en el análisis completo.")

    with tab_features:
        st.markdown("#### Importancia de variables en modelos de árboles")
        importance = result["importance"]
        if importance.empty:
            st.info("No se pudieron calcular importancias para los modelos seleccionados.")
        else:
            st.dataframe(importance, use_container_width=True, hide_index=True)
            plot_cols = [c for c in [MODEL_RF, MODEL_GB] if c in importance.columns]
            for model_col in plot_cols:
                chart_data = importance[["Variable", model_col]].sort_values(model_col, ascending=True).tail(20)
                fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(chart_data))))
                ax.barh(chart_data["Variable"], chart_data[model_col])
                ax.set_title(f"Importancia agregada · {model_col}")
                ax.set_xlabel("Importancia")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        st.info("La importancia mide cuánto contribuye una variable al modelo; no demuestra causalidad. El informe original destaca Support_Calls, Total_Spend, Payment_Delay y Contract_Length.")

    with tab_predictions:
        predictions = result["predictions"]
        st.markdown("#### Predicciones individuales sobre el CSV cargado")
        risk_filter = st.multiselect("Filtrar por nivel de riesgo", ["Alto", "Medio", "Bajo"], default=["Alto", "Medio"], key="risk_filter")
        if risk_filter:
            shown = predictions[predictions["Nivel_riesgo"].isin(risk_filter)]
        else:
            shown = predictions
        st.dataframe(shown.head(1000), use_container_width=True, hide_index=True)
        if len(shown) > 1000:
            st.caption(f"Se muestran las primeras 1.000 de {len(shown):,} filas coincidentes; la descarga incluye todas.")
        st.markdown("**Cómo interpretar los campos**")
        st.write("`Probabilidad_churn_%` es la probabilidad estimada por el modelo; `Nivel_riesgo` usa umbrales operativos de 70 % (alto) y 40 % (medio). `Alerta_retencion` aplica la recomendación del informe cuando reconoce columnas de llamadas a soporte (>3) o retraso en pagos (>0). Ajusta los umbrales a la política de tu empresa antes de usarlos en producción.")
        st.warning("Si el CSV ya contiene el churn real, las predicciones de estas mismas filas son para explorar el riesgo y no sustituyen las métricas de test. Para evaluar rendimiento, consulta la pestaña 'Modelos y métricas'.")

    with tab_export:
        st.markdown("#### Descargar los resultados")
        excel_bytes = make_excel_report(result)
        st.download_button("Descargar informe completo en Excel", data=excel_bytes, file_name="informe_prediccion_churn.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.download_button("Descargar predicciones en CSV", data=frame_to_csv_bytes(result["predictions"]), file_name="predicciones_churn.csv", mime="text/csv")
        st.caption("El Excel contiene Resumen, Churn por tarifa, Churn por contrato, Descriptivos, Correlación, Métricas de modelos, Importancia de variables, Robustez sin contrato y Predicciones.")

    st.divider()
    st.header("4. Predecir clientes nuevos (opcional)")
    st.write("Carga otro CSV sin necesidad de que tenga el churn real. Puedes asignar cada variable del entrenamiento a una cabecera distinta del archivo nuevo.")
    new_file = st.file_uploader("CSV de clientes a puntuar", type=["csv"], key=f"new_csv_{source_fingerprint}")
    if new_file is not None:
        try:
            new_bytes = new_file.getvalue()
            new_df = read_csv_file(io.BytesIO(new_bytes))
            st.caption(f"Archivo: {new_file.name} · {len(new_df):,} filas · {len(new_df.columns)} columnas")
            mapping: dict[str, str] = {}
            with st.form("new_mapping_form"):
                st.markdown("**Mapeo de las cabeceras del CSV nuevo**")
                grid_cols = st.columns(2)
                for idx, feature in enumerate(result["feature_cols"]):
                    options = list(new_df.columns)
                    default_source = feature if feature in options else None
                    selected_idx = options.index(default_source) if default_source in options else 0
                    mapping[feature] = grid_cols[idx % 2].selectbox(f"{feature} ← cabecera del CSV nuevo", options, index=selected_idx, key=f"new_map_{source_fingerprint}_{new_file.name}_{feature}")
                id_new_options = [no_use] + list(new_df.columns)
                id_suggested_new = find_column(new_df.columns, [result["id_col"]] if result["id_col"] else ["CustomerID", "ID Cliente", "id"])
                id_new_default = id_suggested_new if id_suggested_new in id_new_options else no_use
                id_new_source = st.selectbox("Identificador que se conservará en las predicciones (opcional)", id_new_options, index=id_new_options.index(id_new_default), key=f"new_id_{source_fingerprint}_{new_file.name}")
                chosen_model = st.selectbox("Modelo que se usará para los clientes nuevos", MODEL_ORDER, index=MODEL_ORDER.index(result["best_model_name"]), key=f"new_model_{source_fingerprint}_{new_file.name}")
                predict_clicked = st.form_submit_button("Predecir churn de los nuevos clientes", type="primary")
            new_prediction_signature = f"{source_fingerprint}:{new_file.name}:{chosen_model}:{json.dumps(mapping, sort_keys=True)}:{id_new_source}"
            if predict_clicked:
                try:
                    new_predictions = predict_new_customers(
                        new_df,
                        result["production_models"][chosen_model],
                        result["feature_cols"],
                        result["numeric_cols"],
                        mapping,
                        None if id_new_source == no_use else id_new_source,
                    )
                    st.session_state["new_predictions"] = new_predictions
                    st.session_state["new_predictions_signature"] = new_prediction_signature
                except Exception:
                    st.error("No se pudo generar la predicción. Comprueba que las columnas seleccionadas coincidan con las variables usadas al entrenar el modelo.")
            new_predictions = st.session_state.get("new_predictions")
            predictions_are_current = st.session_state.get("new_predictions_signature") == new_prediction_signature
            if new_predictions is not None and predictions_are_current:
                st.markdown("#### Resultado de clientes nuevos")
                nr1, nr2, nr3 = st.columns(3)
                nr1.metric("Clientes puntuados", f"{len(new_predictions):,}")
                nr2.metric("Riesgo alto", f"{(new_predictions['Nivel_riesgo'] == 'Alto').sum():,}")
                nr3.metric("Alertas prioritarias", f"{(new_predictions['Alerta_retencion'] == 'Prioritaria').sum():,}")
                st.dataframe(new_predictions.head(1000), use_container_width=True, hide_index=True)
                st.download_button("Descargar predicciones de clientes nuevos", data=frame_to_csv_bytes(new_predictions), file_name="predicciones_clientes_nuevos.csv", mime="text/csv", key=f"download_new_{source_fingerprint}_{new_file.name}")
        except Exception:
            st.error("No se pudo leer el CSV nuevo. Comprueba el tamaño, la codificación, las cabeceras y el formato del archivo.")

st.divider()
st.markdown("<p class='small-note'>Herramienta de apoyo analítico. Los resultados dependen de la calidad, representatividad y definición de las etiquetas del CSV. Revisa las variables y las alertas antes de adoptar decisiones comerciales.</p>", unsafe_allow_html=True)
