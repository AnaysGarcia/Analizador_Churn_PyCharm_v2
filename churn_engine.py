"""Motor analítico para el proyecto de predicción de churn de Tokio Telecom.

La capa de interfaz (app.py) usa estas funciones para que el análisis también
pueda probarse de forma independiente de Streamlit.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_RF = "Random Forest"
MODEL_LR = "Regresión Logística"
MODEL_GB = "Gradient Boosting"
MODEL_ORDER = [MODEL_RF, MODEL_LR, MODEL_GB]

# Límites defensivos para reducir el riesgo de archivos accidentales o abusivos.
MAX_CSV_BYTES = 20 * 1024 * 1024  # 20 MiB
MAX_CSV_ROWS = 200_000
MAX_CSV_COLUMNS = 200


def _check_csv_size(content: bytes) -> None:
    """Rechaza archivos excesivos antes de entregarlos al parser de pandas."""
    if not content:
        raise ValueError("El archivo está vacío.")
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("El CSV supera el límite de 20 MiB. Divide el archivo y vuelve a intentarlo.")


def _check_duplicate_raw_headers(content: bytes) -> None:
    """Detecta cabeceras duplicadas antes de que pandas las renombre automáticamente."""
    sample = content[:128 * 1024]
    text = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("No se reconoce la codificación del CSV.")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    try:
        headers = next(csv.reader(io.StringIO(text), delimiter=delimiter))
    except (csv.Error, StopIteration):
        raise ValueError("No se pudo interpretar la fila de cabeceras del CSV.") from None
    cleaned = [str(header).strip().replace("\ufeff", "") for header in headers]
    nonempty = [header for header in cleaned if header]
    duplicates = sorted({header for header in nonempty if nonempty.count(header) > 1})
    if duplicates:
        # No se incluye contenido de filas ni valores de clientes en el mensaje.
        raise ValueError("Hay cabeceras duplicadas. Renómbralas para que cada columna tenga un nombre único.")
    if len(cleaned) > MAX_CSV_COLUMNS:
        raise ValueError(f"El CSV supera el máximo permitido de {MAX_CSV_COLUMNS} columnas.")


@dataclass
class AnalysisConfig:
    target_col: str
    positive_value: str
    id_col: str | None = None
    subscription_col: str | None = None
    contract_col: str | None = None
    categorical_cols: list[str] | None = None
    numeric_cols: list[str] | None = None
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 3
    optimize_hyperparameters: bool = True


def normalize_header(value: Any) -> str:
    """Normaliza cabeceras para detectar equivalencias sin modificar el CSV."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _label_token(value: Any) -> str:
    token = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    token = token.strip().lower()
    try:
        numeric = float(token.replace(",", "."))
        if np.isfinite(numeric):
            return str(numeric)
    except (ValueError, TypeError):
        pass
    return token


def find_column(columns: list[str] | pd.Index, aliases: list[str]) -> str | None:
    """Busca primero una coincidencia exacta normalizada y luego una parcial."""
    names = list(columns)
    normalized = {name: normalize_header(name) for name in names}
    wanted = [normalize_header(alias) for alias in aliases]
    for alias in wanted:
        for name in names:
            if normalized[name] == alias:
                return name
    for alias in wanted:
        if not alias:
            continue
        candidates = [name for name in names if alias in normalized[name] or normalized[name] in alias]
        if candidates:
            return min(candidates, key=lambda name: abs(len(normalized[name]) - len(alias)))
    return None


def suggest_column_mappings(columns: list[str] | pd.Index) -> dict[str, str | None]:
    """Sugerencias iniciales para mapear el CSV a los conceptos del informe."""
    return {
        "id": find_column(columns, ["CustomerID", "Customer ID", "ID Cliente", "Cliente ID", "id"]),
        "target": find_column(columns, ["Churn_YesNo", "Churn", "Abandono", "Baja", "Desercion"]),
        "subscription": find_column(columns, ["Subscription_Type", "Subscription Type", "Tipo de tarifa", "Tarifa", "Plan"]),
        "contract": find_column(columns, ["Contract_Length", "Contract Length", "Duracion del contrato", "Contrato"]),
    }


def read_csv_file(file_obj: Any) -> pd.DataFrame:
    """Lee y valida un CSV local, con separador automático y codificaciones comunes.

    Aplica límites de tamaño, filas y columnas, detecta cabeceras duplicadas y
    rechaza filas mal formadas en lugar de descartarlas silenciosamente.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    content = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("El archivo debe proporcionarse como bytes o como un archivo binario.")
    content = bytes(content)
    _check_csv_size(content)
    _check_duplicate_raw_headers(content)

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(
                io.BytesIO(content), encoding=encoding, sep=None, engine="python",
                on_bad_lines="error",
            )
            frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame.columns]
            # Elimina columnas vacías que algunos exportadores añaden al final.
            frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed:\s*\d+$")]
            if frame.empty and len(frame.columns) == 0:
                raise ValueError("El archivo no contiene columnas reconocibles.")
            if frame.columns.duplicated().any():
                raise ValueError("Hay cabeceras duplicadas. Renómbralas para que cada columna tenga un nombre único.")
            if len(frame.columns) > MAX_CSV_COLUMNS:
                raise ValueError(f"El CSV supera el máximo permitido de {MAX_CSV_COLUMNS} columnas.")
            if len(frame) > MAX_CSV_ROWS:
                raise ValueError(f"El CSV supera el máximo permitido de {MAX_CSV_ROWS} filas de datos.")
            if len(frame.columns) == 0 or any(not str(c).strip() for c in frame.columns):
                raise ValueError("Todas las columnas deben tener una cabecera no vacía.")
            return frame
        except ValueError:
            raise
        except Exception as exc:  # probar otras codificaciones antes de informar
            last_error = exc
    # No incluir el texto de la excepción de pandas: podría contener datos del archivo.
    raise ValueError("No se pudo leer el CSV. Comprueba la codificación, el separador, las cabeceras y que las filas estén bien formadas.") from last_error


def _parse_numeric_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return np.nan
    text = text.replace("\u00a0", "").replace(" ", "")
    text = re.sub(r"[€$£%]", "", text)
    # Si hay ambos separadores, el último se interpreta como decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convierte una variable a número, contemplando decimales con coma."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    return series.map(_parse_numeric_value).astype(float)


def infer_feature_columns(
    frame: pd.DataFrame,
    target_col: str | None = None,
    id_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """Infiere variables categóricas y numéricas, ignorando el objetivo e ID."""
    categorical: list[str] = []
    numeric: list[str] = []
    excluded = {x for x in (target_col, id_col) if x}
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            candidate = coerce_numeric_series(series)
            non_null = series.notna().sum()
            ratio = candidate.notna().sum() / max(non_null, 1)
            if non_null and ratio >= 0.90 and series.nunique(dropna=True) > 8:
                numeric.append(column)
            else:
                categorical.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            # Una columna numérica con pocos valores únicos puede representar
            # una categoría codificada (por ejemplo, 1/2/3), pero no se fuerza.
            numeric.append(column)
        else:
            categorical.append(column)
    return categorical, numeric


def _make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if not transformers:
        raise ValueError("Selecciona al menos una variable predictora, categórica o numérica.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _model_specs(optimize: bool) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    if optimize:
        return {
            MODEL_RF: (
                RandomForestClassifier(random_state=42, n_jobs=1),
                {"classifier__max_depth": [5, 10], "classifier__n_estimators": [50, 100]},
            ),
            MODEL_LR: (
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=42),
                {"classifier__C": [0.1, 1.0, 10.0]},
            ),
            MODEL_GB: (
                GradientBoostingClassifier(random_state=42),
                {"classifier__max_depth": [3, 5], "classifier__n_estimators": [50, 100]},
            ),
        }
    return {
        MODEL_RF: (RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1), {}),
        MODEL_LR: (LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42), {}),
        MODEL_GB: (GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3), {}),
    }


def _metrics_for_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_array = np.asarray(y_true, dtype=int)
    cm = confusion_matrix(y_array, np.asarray(y_pred, dtype=int), labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    if len(np.unique(y_array)) > 1:
        auc = float(roc_auc_score(y_array, probability))
        pr_auc = float(average_precision_score(y_array, probability))
    else:
        auc, pr_auc = np.nan, np.nan
    return {
        "AUC-ROC": auc,
        "AUC-PR": pr_auc,
        "Accuracy": float(accuracy_score(y_array, y_pred)),
        "Precision (churn)": float(precision_score(y_array, y_pred, zero_division=0)),
        "Recall (churn)": float(recall_score(y_array, y_pred, zero_division=0)),
        "Especificidad": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1 (churn)": float(f1_score(y_array, y_pred, zero_division=0)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def _score_frame(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predicted = model.predict(x).astype(int)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(model.named_steps["classifier"].classes_)
        positive_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
        probability = probabilities[:, positive_idx]
    else:
        # Los tres clasificadores utilizados tienen predict_proba; este bloque
        # queda como protección ante un cambio de algoritmo.
        probability = predicted.astype(float)
    return predicted, np.asarray(probability, dtype=float)


def _aggregate_feature_importances(model: Pipeline, categorical_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    classifier = model.named_steps.get("classifier")
    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["Variable", "Importancia"])
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = np.asarray(classifier.feature_importances_, dtype=float)
    source_features = list(numeric_cols) + list(categorical_cols)
    totals = {name: 0.0 for name in source_features}
    for transformed_name, importance in zip(names, values):
        clean_name = re.sub(r"^(num|cat)__", "", str(transformed_name))
        # La categoría se agrega a la variable original. El orden descendente
        # por longitud evita confundir nombres como Usage y Usage_Frequency.
        matches = [col for col in source_features if clean_name == col or clean_name.startswith(col + "_")]
        if matches:
            totals[max(matches, key=len)] += float(importance)
    result = pd.DataFrame({"Variable": list(totals), "Importancia": list(totals.values())})
    return result.sort_values("Importancia", ascending=False).reset_index(drop=True)


def _segment_summary(frame: pd.DataFrame, segment_col: str | None, target_binary: pd.Series) -> pd.DataFrame:
    if not segment_col or segment_col not in frame.columns:
        return pd.DataFrame(columns=["Segmento", "Clientes", "Churn", "Churn (%)"])
    temp = pd.DataFrame({"Segmento": frame[segment_col].astype("string").fillna("(Sin dato)"), "ChurnBinario": target_binary})
    summary = temp.groupby("Segmento", dropna=False).agg(Clientes=("ChurnBinario", "size"), Churn=("ChurnBinario", "sum"))
    summary["Churn (%)"] = (summary["Churn"] / summary["Clientes"] * 100).round(2)
    return summary.reset_index()


def _descriptive_stats(frame: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    numeric = pd.DataFrame({col: coerce_numeric_series(frame[col]) for col in numeric_cols})
    table = numeric.describe().T.rename_axis("Variable").reset_index()
    return table.round(4)


def _correlation_matrix(frame: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    numeric = pd.DataFrame({col: coerce_numeric_series(frame[col]) for col in numeric_cols})
    return numeric.corr(numeric_only=True).round(4)


def _risk_prediction_frame(
    frame: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    id_col: str | None,
    support_calls_col: str | None,
    payment_delay_col: str | None,
    real_target_col: str | None = None,
) -> pd.DataFrame:
    x = frame[feature_cols].copy()
    predicted, probability = _score_frame(model, x)
    result = pd.DataFrame(index=frame.index)
    if id_col and id_col in frame.columns:
        result[id_col] = frame[id_col]
    if real_target_col and real_target_col in frame.columns:
        result["Churn_real"] = frame[real_target_col]
    result["Probabilidad_churn_%"] = (probability * 100).round(2)
    result["Prediccion_churn"] = np.where(predicted == 1, "Yes", "No")
    result["Nivel_riesgo"] = np.select(
        [probability >= 0.70, probability >= 0.40],
        ["Alto", "Medio"],
        default="Bajo",
    )
    operational_flag = pd.Series(False, index=frame.index)
    operational_rule_available = False
    if support_calls_col and support_calls_col in frame.columns:
        calls = coerce_numeric_series(frame[support_calls_col])
        operational_flag |= calls > 3
        operational_rule_available = True
    if payment_delay_col and payment_delay_col in frame.columns:
        delays = coerce_numeric_series(frame[payment_delay_col])
        operational_flag |= delays > 0
        operational_rule_available = True
    if operational_rule_available:
        result["Alerta_retencion"] = np.where(operational_flag, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))
    else:
        result["Alerta_retencion"] = np.where(probability >= 0.70, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))
    return result.reset_index(drop=True)


def run_analysis(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Ejecuta el análisis descriptivo, comparación de modelos y predicciones.

    La evaluación de rendimiento se calcula en un conjunto de prueba reservado.
    Después de evaluar, cada mejor pipeline se vuelve a ajustar con todos los
    registros etiquetados para disponer de modelos finales que puntúen clientes.
    """
    if config.target_col not in frame.columns:
        raise ValueError("La columna objetivo seleccionada no existe en el CSV.")
    if frame.empty:
        raise ValueError("El CSV está vacío.")

    work = frame.copy().reset_index(drop=True)
    categorical_cols = list(dict.fromkeys(config.categorical_cols or []))
    numeric_cols = list(dict.fromkeys(config.numeric_cols or []))
    # Evitar fuga de información: el ID y la variable objetivo nunca se usan como predictores.
    forbidden = {config.target_col}
    if config.id_col:
        forbidden.add(config.id_col)
    categorical_cols = [c for c in categorical_cols if c in work.columns and c not in forbidden]
    numeric_cols = [c for c in numeric_cols if c in work.columns and c not in forbidden and c not in categorical_cols]
    feature_cols = categorical_cols + numeric_cols
    if not feature_cols:
        raise ValueError("No hay variables predictoras. Selecciona campos categóricos o numéricos distintos del ID y del churn.")

    label_strings = work[config.target_col].astype("string")
    positive_token = _label_token(config.positive_value)
    target_valid = work[config.target_col].notna() & label_strings.str.strip().ne("")
    y_full = label_strings.map(lambda val: 1 if _label_token(val) == positive_token else 0).astype(int)
    labeled_idx = np.flatnonzero(target_valid.to_numpy())
    if len(labeled_idx) < 10:
        raise ValueError("Se necesitan al menos 10 filas con una etiqueta de churn válida para entrenar y evaluar.")
    X_all_labeled = work.loc[labeled_idx, feature_cols].copy().reset_index(drop=True)
    y_all_labeled = y_full.iloc[labeled_idx].reset_index(drop=True)
    class_counts = y_all_labeled.value_counts()
    if len(class_counts) != 2:
        raise ValueError("La columna objetivo debe contener al menos un caso de churn y uno de no churn.")
    if int(class_counts.min()) < 2:
        raise ValueError("Se necesitan al menos dos ejemplos de cada clase (Yes/No) para dividir los datos de forma estratificada.")

    # Conversiones numéricas sin alterar las cabeceras del archivo.
    for col in numeric_cols:
        X_all_labeled[col] = coerce_numeric_series(X_all_labeled[col])
    x_train, x_test, y_train, y_test = train_test_split(
        X_all_labeled,
        y_all_labeled,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_all_labeled,
    )
    cv_folds = min(max(2, int(config.cv_folds)), int(y_train.value_counts().min()))
    if cv_folds < 2:
        raise ValueError("Hay muy pocos ejemplos por clase para aplicar validación cruzada.")

    specs = _model_specs(config.optimize_hyperparameters)
    metrics_rows: list[dict[str, Any]] = []
    fitted_for_evaluation: dict[str, Pipeline] = {}
    best_params: dict[str, dict[str, Any]] = {}
    cv_scores: dict[str, float] = {}
    roc_test: dict[str, dict[str, np.ndarray | float]] = {}
    importance_frames: list[pd.DataFrame] = []
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)

    for index, model_name in enumerate(MODEL_ORDER, start=1):
        estimator, grid = specs[model_name]
        preprocessor = _make_preprocessor(categorical_cols, numeric_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        if config.optimize_hyperparameters:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                scoring="roc_auc",
                cv=splitter,
                refit=True,
                n_jobs=2,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                search.fit(x_train, y_train)
            evaluation_model = search.best_estimator_
            params = search.best_params_
            mean_cv_auc = float(search.best_score_)
        else:
            params = {}
            # Sin Grid Search se conservan los parámetros base, pero se calcula
            # AUC de validación cruzada para seguir comparando sin usar el test.
            cv_values = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=splitter, n_jobs=2)
            mean_cv_auc = float(np.mean(cv_values))
            evaluation_model = pipeline.fit(x_train, y_train)
        fitted_for_evaluation[model_name] = evaluation_model
        best_params[model_name] = params
        cv_scores[model_name] = mean_cv_auc
        for subset_name, x_subset, y_subset in [("Train", x_train, y_train), ("Test", x_test, y_test)]:
            predicted, probability = _score_frame(evaluation_model, x_subset)
            row = {"Modelo": model_name, "Conjunto": subset_name, **_metrics_for_predictions(y_subset, predicted, probability)}
            metrics_rows.append(row)
            if subset_name == "Test":
                # Coordenadas ROC se calculan en app.py, usando y_test/predicción guardados.
                roc_test[model_name] = {"y_true": np.asarray(y_subset, dtype=int), "probability": probability}
        if model_name in (MODEL_RF, MODEL_GB):
            fi = _aggregate_feature_importances(evaluation_model, categorical_cols, numeric_cols)
            if not fi.empty:
                fi = fi.rename(columns={"Importancia": model_name})
                importance_frames.append(fi)
        if progress_callback:
            progress_callback(index, len(MODEL_ORDER), model_name)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["CV AUC medio"] = metrics_df["Modelo"].map(cv_scores)
    metrics_df = metrics_df[["Modelo", "Conjunto", "CV AUC medio", "AUC-ROC", "AUC-PR", "Accuracy", "Precision (churn)", "Recall (churn)", "Especificidad", "F1 (churn)", "TN", "FP", "FN", "TP"]]

    # Robustez opcional descrita en la Tarea 7: volver a entrenar con los
    # mejores hiperparámetros, pero sin Contract_Length, y comparar el AUC test.
    robustness = pd.DataFrame()
    contract_feature = config.contract_col if config.contract_col in feature_cols else None
    if contract_feature and len(feature_cols) > 1:
        reduced_cat = [c for c in categorical_cols if c != contract_feature]
        reduced_num = [c for c in numeric_cols if c != contract_feature]
        reduced_features = reduced_cat + reduced_num
        x_train_reduced = x_train[reduced_features].copy()
        x_test_reduced = x_test[reduced_features].copy()
        if reduced_features:
            robustness_rows = []
            for model_name in MODEL_ORDER:
                reduced_classifier = clone(fitted_for_evaluation[model_name].named_steps["classifier"])
                tuned_params = {key.replace("classifier__", "", 1): value for key, value in best_params[model_name].items()}
                if tuned_params:
                    reduced_classifier.set_params(**tuned_params)
                reduced_pipeline = Pipeline([
                    ("preprocessor", _make_preprocessor(reduced_cat, reduced_num)),
                    ("classifier", reduced_classifier),
                ])
                reduced_pipeline.fit(x_train_reduced, y_train)
                _, reduced_probability = _score_frame(reduced_pipeline, x_test_reduced)
                full_auc = float(metrics_df[(metrics_df["Modelo"] == model_name) & (metrics_df["Conjunto"] == "Test")]["AUC-ROC"].iloc[0])
                without_auc = float(roc_auc_score(y_test, reduced_probability))
                robustness_rows.append({
                    "Modelo": model_name,
                    "AUC-ROC completo (test)": full_auc,
                    "AUC-ROC sin contrato (test)": without_auc,
                    "Caída de AUC": full_auc - without_auc,
                })
            robustness = pd.DataFrame(robustness_rows)

    # Modelo recomendado: mayor AUC medio de validación cruzada; no se elige mirando el test.
    best_model_name = max(cv_scores, key=cv_scores.get)
    # Ajuste de producción con todos los registros etiquetados. La tabla de
    # métricas anterior sigue correspondiendo exclusivamente a train/test.
    production_models: dict[str, Pipeline] = {}
    for model_name, model in fitted_for_evaluation.items():
        production_models[model_name] = clone(model).fit(X_all_labeled, y_all_labeled)
    production_model = production_models[best_model_name]

    # Tabla de predicciones para todas las filas del histórico (también permite
    # detectar filas cuya etiqueta esté vacía). Las métricas son las del holdout.
    X_all = work[feature_cols].copy()
    for col in numeric_cols:
        X_all[col] = coerce_numeric_series(X_all[col])
    historical_output = work.copy()
    predicted, probability = _score_frame(production_model, X_all)
    historical_output["Probabilidad_churn_%"] = (probability * 100).round(2)
    historical_output["Prediccion_churn"] = np.where(predicted == 1, "Yes", "No")
    historical_output["Nivel_riesgo"] = np.select(
        [probability >= 0.70, probability >= 0.40], ["Alto", "Medio"], default="Bajo"
    )
    support_calls_col = find_column(work.columns, ["Support_Calls", "Support Calls", "Llamadas_soporte", "Llamadas a soporte", "Llamadas_soporte_tecnico"])
    payment_delay_col = find_column(work.columns, ["Payment_Delay", "Payment Delay", "Retraso_pago", "Retrasos en pagos", "Dias_retraso"])
    if support_calls_col or payment_delay_col:
        alert = pd.Series(False, index=work.index)
        if support_calls_col:
            alert |= coerce_numeric_series(work[support_calls_col]) > 3
        if payment_delay_col:
            alert |= coerce_numeric_series(work[payment_delay_col]) > 0
        historical_output["Alerta_retencion"] = np.where(alert, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))
    else:
        historical_output["Alerta_retencion"] = np.where(probability >= 0.70, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))

    valid_original_indices = np.flatnonzero(target_valid.to_numpy())
    target_for_segments = y_full.iloc[valid_original_indices].reset_index(drop=True)
    labeled_work = work.iloc[valid_original_indices].reset_index(drop=True)
    by_subscription = _segment_summary(labeled_work, config.subscription_col, target_for_segments)
    by_contract = _segment_summary(labeled_work, config.contract_col, target_for_segments)

    numeric_for_summary = list(dict.fromkeys(numeric_cols))
    descriptive = _descriptive_stats(work, numeric_for_summary)
    correlation = _correlation_matrix(work, numeric_for_summary)

    # Importancia agregada por variable original para cada modelo de árbol.
    if importance_frames:
        importance = importance_frames[0]
        for other in importance_frames[1:]:
            importance = importance.merge(other, on="Variable", how="outer")
        importance = importance.fillna(0).sort_values(MODEL_RF if MODEL_RF in importance.columns else importance.columns[1], ascending=False)
    else:
        importance = pd.DataFrame(columns=["Variable", MODEL_RF, MODEL_GB])

    # Medidas de calidad del target en el CSV, calculadas sobre filas etiquetadas.
    total_labeled = int(target_valid.sum())
    churn_rate = float(y_full[target_valid].mean() * 100) if total_labeled else np.nan
    summary = {
        "Filas CSV": int(len(work)),
        "Columnas CSV": int(len(work.columns)),
        "Filas con churn conocido": total_labeled,
        "Clientes churn (etiqueta positiva)": int(y_full[target_valid].sum()),
        "Tasa churn real (%)": round(churn_rate, 2),
        "Variables predictoras": ", ".join(feature_cols),
        "Columna objetivo": config.target_col,
        "Valor que representa churn": config.positive_value,
        "Separación train/test": f"{(1-config.test_size)*100:.0f}/{config.test_size*100:.0f}",
        "Semilla": config.random_state,
        "Validación cruzada": f"{cv_folds} folds" if config.optimize_hyperparameters else f"{cv_folds} folds; hiperparámetros por defecto",
        "Modelo recomendado por AUC CV": best_model_name,
        "AUC CV del modelo recomendado": round(cv_scores[best_model_name], 4),
        "Nota": "Las métricas de calidad se calculan en el conjunto de prueba reservado; las predicciones del CSV completo se generan con el modelo final ajustado con todos los registros etiquetados.",
    }

    test_indices = x_test.index.to_numpy()
    test_actual = y_test.reset_index(drop=True)
    roc_curves = {}
    from sklearn.metrics import roc_curve
    for model_name, item in roc_test.items():
        fpr, tpr, _ = roc_curve(item["y_true"], item["probability"])
        roc_curves[model_name] = {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(item["y_true"], item["probability"]))}

    return {
        "summary": summary,
        "metrics": metrics_df,
        "by_subscription": by_subscription,
        "by_contract": by_contract,
        "descriptive": descriptive,
        "correlation": correlation,
        "importance": importance,
        "robustness": robustness,
        "predictions": historical_output,
        "models": fitted_for_evaluation,
        "production_models": production_models,
        "best_model_name": best_model_name,
        "best_model": production_model,
        "best_params": best_params,
        "cv_scores": cv_scores,
        "roc_curves": roc_curves,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_binary": y_full,
        "target_valid": target_valid,
        "target_col": config.target_col,
        "id_col": config.id_col,
        "support_calls_col": support_calls_col,
        "payment_delay_col": payment_delay_col,
        "cv_folds_used": cv_folds,
        "test_size": config.test_size,
    }


def predict_new_customers(
    new_frame: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    numeric_cols: list[str],
    column_mapping: dict[str, str],
    id_source_col: str | None = None,
) -> pd.DataFrame:
    """Predice churn para un CSV nuevo con cabeceras distintas si se mapean."""
    missing = [feature for feature in feature_cols if feature not in column_mapping or column_mapping[feature] not in new_frame.columns]
    if missing:
        raise ValueError("Falta asignar una cabecera del CSV nuevo a estas variables: " + ", ".join(missing))
    x = pd.DataFrame(index=new_frame.index)
    for feature in feature_cols:
        x[feature] = new_frame[column_mapping[feature]]
        if feature in numeric_cols:
            x[feature] = coerce_numeric_series(x[feature])
    predicted, probability = _score_frame(model, x)
    output = pd.DataFrame(index=new_frame.index)
    if id_source_col and id_source_col in new_frame.columns:
        output[id_source_col] = new_frame[id_source_col]
    output["Probabilidad_churn_%"] = (probability * 100).round(2)
    output["Prediccion_churn"] = np.where(predicted == 1, "Yes", "No")
    output["Nivel_riesgo"] = np.select([probability >= 0.70, probability >= 0.40], ["Alto", "Medio"], default="Bajo")
    support_training_col = find_column(feature_cols, ["Support_Calls", "Support Calls", "Llamadas a soporte", "Llamadas_soporte", "Llamadas_soporte_tecnico"])
    delay_training_col = find_column(feature_cols, ["Payment_Delay", "Payment Delay", "Retrasos en pagos", "Retraso_pago", "Dias_retraso"])
    support = column_mapping.get(support_training_col) if support_training_col else None
    delay = column_mapping.get(delay_training_col) if delay_training_col else None
    if support not in new_frame.columns:
        support = find_column(new_frame.columns, ["Support_Calls", "Support Calls", "Llamadas a soporte", "Llamadas_soporte", "Llamadas_soporte_tecnico"])
    if delay not in new_frame.columns:
        delay = find_column(new_frame.columns, ["Payment_Delay", "Payment Delay", "Retrasos en pagos", "Retraso_pago", "Dias_retraso"])
    urgent = pd.Series(False, index=new_frame.index)
    found_rule = False
    if support:
        urgent |= coerce_numeric_series(new_frame[support]) > 3
        found_rule = True
    if delay:
        urgent |= coerce_numeric_series(new_frame[delay]) > 0
        found_rule = True
    if found_rule:
        output["Alerta_retencion"] = np.where(urgent, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))
    else:
        output["Alerta_retencion"] = np.where(probability >= 0.70, "Prioritaria", np.where(probability >= 0.40, "Revisar riesgo", "Sin alerta"))
    return output.reset_index(drop=True)


def _escape_spreadsheet_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Neutraliza texto que hojas de cálculo podrían interpretar como fórmula.

    Solo modifica cadenas de texto; las columnas numéricas conservan su tipo.
    Se aplica únicamente a los archivos exportados, no al análisis en memoria.
    """
    safe = frame.copy()
    dangerous = ("=", "+", "-", "@")
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            def escape(value: Any) -> Any:
                if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(dangerous):
                    return "'" + value
                return value
            safe[column] = safe[column].map(escape)
    return safe


def make_excel_report(result: dict[str, Any]) -> bytes:
    """Crea un Excel descargable y neutraliza fórmulas potenciales en texto."""
    buffer = io.BytesIO()
    summary_df = pd.DataFrame([{"Concepto": k, "Valor": v} for k, v in result["summary"].items()])
    sheets = [
        ("Resumen", summary_df),
        ("Churn_por_tarifa", result["by_subscription"]),
        ("Churn_por_contrato", result["by_contract"]),
        ("Descriptivos", result["descriptive"]),
        ("Correlacion", result["correlation"].reset_index() if not result["correlation"].empty else result["correlation"]),
        ("Metricas_modelos", result["metrics"]),
        ("Importancia_variables", result["importance"]),
        ("Robustez_sin_contrato", result.get("robustness", pd.DataFrame())),
        ("Predicciones", result["predictions"]),
    ]
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, table in sheets:
            if table is None or not isinstance(table, pd.DataFrame):
                table = pd.DataFrame()
            table = _escape_spreadsheet_formulas(table)
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.sheets[sheet_name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [len(str(cell.value)) for cell in list(column_cells)[:200] if cell.value is not None]
                width = min(max(max(values, default=10) + 2, 10), 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return buffer.getvalue()


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Exporta un CSV UTF-8 con BOM y protección frente a fórmulas en hojas de cálculo."""
    safe = _escape_spreadsheet_formulas(frame)
    return safe.to_csv(index=False).encode("utf-8-sig")
