import io
import unittest

import numpy as np
import pandas as pd

from churn_engine import (
    AnalysisConfig,
    coerce_numeric_series,
    find_column,
    infer_feature_columns,
    predict_new_customers,
    read_csv_file,
    run_analysis,
    suggest_column_mappings,
    frame_to_csv_bytes,
    MAX_CSV_BYTES,
    _escape_spreadsheet_formulas,
)


class ChurnEngineTests(unittest.TestCase):
    def make_data(self, n=240):
        rng = np.random.default_rng(7)
        contract = rng.choice(["Monthly", "Annual"], n, p=[0.45, 0.55])
        support = rng.integers(0, 8, n)
        delay = rng.integers(0, 18, n)
        churn_prob = np.clip(0.08 + 0.08 * support + 0.02 * delay, 0.05, 0.95)
        target = np.where(contract == "Monthly", "Yes", np.where(rng.random(n) < churn_prob, "Yes", "No"))
        return pd.DataFrame({
            "Identificador": np.arange(n) + 1,
            "Edad_cliente": rng.integers(18, 80, n),
            "Genero": rng.choice(["Mujer", "Hombre"], n),
            "Llamadas soporte": support,
            "Retraso pago": delay,
            "Tipo tarifa": rng.choice(["Basic", "Standard", "Premium"], n),
            "Plazo": contract,
            "Resultado baja": target,
        })

    def test_read_csv_detects_separator_and_columns(self):
        text = "cliente;resultado;gasto\n1;Yes;10,5\n2;No;20,0\n"
        frame = read_csv_file(io.BytesIO(text.encode("utf-8")))
        self.assertEqual(list(frame.columns), ["cliente", "resultado", "gasto"])
        self.assertEqual(len(frame), 2)


    def test_duplicate_headers_are_rejected(self):
        text = "cliente,resultado,resultado\n1,Yes,No\n"
        with self.assertRaisesRegex(ValueError, "cabeceras duplicadas"):
            read_csv_file(io.BytesIO(text.encode("utf-8")))

    def test_oversized_csv_is_rejected(self):
        from churn_engine import _check_csv_size
        with self.assertRaisesRegex(ValueError, "20 MiB"):
            _check_csv_size(b"x" * (MAX_CSV_BYTES + 1))

    def test_spreadsheet_escape_keeps_numeric_values(self):
        frame = pd.DataFrame({"texto": ["@SUM(A1:A2)", "seguro"], "numero": [-12, 4]})
        safe = _escape_spreadsheet_formulas(frame)
        self.assertEqual(safe.loc[0, "texto"], "'@SUM(A1:A2)")
        self.assertEqual(safe.loc[0, "numero"], -12)

    def test_csv_export_neutralizes_formula_text(self):
        frame = pd.DataFrame({"cliente": ["=HYPERLINK(\"https://example.com\")", "normal"], "importe": [12, 13]})
        exported = frame_to_csv_bytes(frame).decode("utf-8-sig")
        self.assertIn("'=HYPERLINK", exported)
        self.assertIn("normal", exported)

    def test_numeric_conversion_supports_decimal_comma(self):
        s = pd.Series(["10,5", "2.000,75", "1,000.25", ""])
        converted = coerce_numeric_series(s)
        self.assertAlmostEqual(converted.iloc[0], 10.5)
        self.assertAlmostEqual(converted.iloc[1], 2000.75)
        self.assertAlmostEqual(converted.iloc[2], 1000.25)
        self.assertTrue(np.isnan(converted.iloc[3]))

    def test_header_suggestions(self):
        mappings = suggest_column_mappings(["ID cliente", "Resultado baja", "Tipo tarifa", "Plazo"])
        self.assertEqual(mappings["target"], "Resultado baja")
        self.assertEqual(mappings["subscription"], "Tipo tarifa")
        self.assertEqual(find_column(["Retraso pago", "Edad"], ["Payment_Delay", "Retraso_pago"]), "Retraso pago")

    def test_end_to_end_without_grid_search(self):
        data = self.make_data()
        cat, num = infer_feature_columns(data, target_col="Resultado baja", id_col="Identificador")
        config = AnalysisConfig(
            target_col="Resultado baja",
            positive_value="Yes",
            id_col="Identificador",
            subscription_col="Tipo tarifa",
            contract_col="Plazo",
            categorical_cols=cat,
            numeric_cols=num,
            test_size=0.20,
            cv_folds=3,
            optimize_hyperparameters=False,
        )
        result = run_analysis(data, config)
        self.assertIn("Random Forest", result["metrics"]["Modelo"].unique())
        self.assertEqual(len(result["by_subscription"]), 3)
        self.assertGreater(len(result["predictions"]), 0)
        self.assertIn("Probabilidad_churn_%", result["predictions"].columns)
        self.assertIn("AUC-ROC", result["metrics"].columns)
        self.assertIn("Caída de AUC", result["robustness"].columns)
        self.assertTrue(0.0 <= result["cv_scores"][result["best_model_name"]] <= 1.0)

    def test_predict_new_data_with_mapped_headers(self):
        data = self.make_data()
        cat, num = infer_feature_columns(data, target_col="Resultado baja", id_col="Identificador")
        config = AnalysisConfig(
            target_col="Resultado baja", positive_value="Yes", id_col="Identificador",
            categorical_cols=cat, numeric_cols=num, optimize_hyperparameters=False,
        )
        result = run_analysis(data, config)
        new = pd.DataFrame({
            "ID nuevo": [9001, 9002],
            "edad": [44, 51],
            "genero": ["Mujer", "Hombre"],
            "llamadas": [5, 0],
            "dias de demora": [8, 0],
            "tarifa": ["Basic", "Premium"],
            "contrato": ["Monthly", "Annual"],
        })
        mapping = {
            "Edad_cliente": "edad",
            "Genero": "genero",
            "Llamadas soporte": "llamadas",
            "Retraso pago": "dias de demora",
            "Tipo tarifa": "tarifa",
            "Plazo": "contrato",
        }
        pred = predict_new_customers(new, result["production_models"][result["best_model_name"]], result["feature_cols"], result["numeric_cols"], mapping, "ID nuevo")
        self.assertEqual(len(pred), 2)
        self.assertIn("Nivel_riesgo", pred.columns)
        self.assertIn("Alerta_retencion", pred.columns)


if __name__ == "__main__":
    unittest.main()
