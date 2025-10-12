# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  Imports & Logging
# ──────────────────────────────────────────────────────────────────────────────
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import mlflow
import s3fs
import nltk

# Download the NLTK stop‑words once
nltk.download("stopwords")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Create an s3fs file system that works with MinIO *or* AWS S3."""
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def upload_parquet(df: pd.DataFrame, s3_uri: str) -> None:
    """
    Write *df* to *s3_uri* in Parquet format using the configured S3 file system.
    """
    fs = get_s3_filesystem()
    log.info(f"Uploading dataframe to {s3_uri}")
    df.to_parquet(s3_uri, index=False, filesystem=fs, engine="pyarrow")


# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣  MLflow helpers
# ──────────────────────────────────────────────────────────────────────────────
def download_mlflow_artifact(
    artifact_uri: str, dst_path: Path | str = "./tmp_artifacts"
) -> Path:
    """
    Download artifacts from MLflow into *dst_path* and return the absolute path.
    """
    dst = Path(dst_path).expanduser().resolve()
    log.info(f"Downloading artifact {artifact_uri} to {dst}")
    mlflow.artifacts.download_artifacts(artifact_uri, dst_path=dst)
    return dst


def load_pyfunc_model(model_path: Path | str) -> mlflow.pyfunc.PyFuncModel:
    """Load an MLflow pyfunc model from *model_path*."""
    path = Path(model_path).expanduser()
    log.info(f"Loading pyfunc model from {path}")
    return mlflow.pyfunc.load_model(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_input_items(texts: List[str]) -> List[Dict[str, str]]:
    """
    Convert a list of free‑text strings into the list of dictionaries
    expected by the model. Only the field `description_activity` is required.
    """
    return [{"description_activity": t} for t in texts]


def extract_prediction_fields(preds: List[mlflow.pyfunc.PyFuncModel]) -> List[Dict[str, Any]]:
    """
    Convert the list of `mlflow.pyfunc.PyFuncModel` objects returned by
    `model.predict` into a plain Python list of dictionaries so that
    it can be used like a normal pandas row.
    """
    return [pred.model_dump() for pred in preds]

# ────────────────────────────────────────────────────────────────────────────────
# 5️⃣  Pipeline Golden‑Tests (mis à jour)
# ────────────────────────────────────────────────────────────────────────────────


def run_golden_tests(
    model: mlflow.pyfunc.PyFuncModel,
    golden_csv: str,
    upload_uri: str,
) -> None:
    """Exécute l’ensemble du pipeline de golden‑tests."""
    # --- 1. Chargement des données
    golden = pd.read_csv(golden_csv, encoding="utf8", delimiter=";")
    log.info(f"Loaded {len(golden)} golden test rows")

    # --- 2. Prédictions
    input_items = build_input_items(golden["libelle"].tolist())
    preds = model.predict(input_items)
    pred_dicts = extract_prediction_fields(preds)

    golden["APE_prediction"] = [d["1"]["code"] for d in pred_dicts]
    golden["IC"] = [d["IC"] for d in pred_dicts]

    # ── 3. Statistiques globales ------------------------------
    overall_mean = golden["IC"].mean()
    overall_median = golden["IC"].median()
    log.info(f"Global IC – mean: {overall_mean:.4f}, median: {overall_median:.4f}")

    # ── 4. Statistiques sur les IC par groupe ----------------
    ok_mask = golden["nace2025"] == golden["APE_prediction"]
    err_mask = ~ok_mask

    # Concordants
    ok_ic_mean = golden.loc[ok_mask, "IC"].mean()
    ok_ic_median = golden.loc[ok_mask, "IC"].median()
    log.info(f"Concordant IC – mean: {ok_ic_mean:.4f}, median: {ok_ic_median:.4f}")

    # Non‑concordants
    err_ic_mean = golden.loc[err_mask, "IC"].mean()
    err_ic_median = golden.loc[err_mask, "IC"].median()
    log.info(f"Non‑concordant IC – mean: {err_ic_mean:.4f}, median: {err_ic_median:.4f}")

    # Rapport
    concordance_rate = ok_mask.mean()
    log.info(f"Concordance rate: {concordance_rate:.4f}")
    log.info(f"Overall mean IC: {overall_mean:.4f}")
    log.info(f"Overall median IC: {overall_median:.4f}")

    # ── 5. Split & upload ------------------------------------
    ok_rows = golden[ok_mask]
    err_rows = golden[err_mask]

    upload_parquet(golden, f"{upload_uri}/golden_tests_results.parquet")
    upload_parquet(err_rows, f"{upload_uri}/golden_tests_error.parquet")
    upload_parquet(ok_rows[["nace2025", "libelle", "CRT", "APE_prediction", "IC"]],
                   f"{upload_uri}/golden_tests_ok.parquet")

    log.debug(golden.head())

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1️⃣  Configure & download model ───────────────────────────────────────
    MODEL_URI = "runs:/4f09a147df0f4226ba45b49ae3a5a853/pyfunc_model"
    dst_path = Path("../my_model").expanduser().resolve()
    download_mlflow_artifact(MODEL_URI, dst_path)

    # ── 2️⃣  Add NLTK data folder to the data path ─────────────────────────────
    nltk_data_folder = dst_path / "artifacts" / "nltk_data"
    nltk.data.path.append(str(nltk_data_folder))
    log.info(f"NLTK data path updated: {nltk.data.path}")

    # ── 3️⃣  Load the model ───────────────────────────────────────────────────
    model = load_pyfunc_model(dst_path / "pyfunc_model")

    # ── 4️⃣  Run golden‑tests pipeline ─────────────────────────────────────────
    run_golden_tests(
        model=model,
        golden_csv="golden_tests.csv",
        upload_uri="s3://projet-ape/data",
    )

    # ── 5️⃣  Additional predictions on a second CSV ────────────────────────────
    df = pd.read_csv("resultats_comparaison_ape.csv", encoding="utf8", delimiter=" ")
    texts = df["Texte_Descriptif"].tolist()

    preds = model.predict(build_input_items(texts))
    pred_dicts = extract_prediction_fields(preds)

    df["Predicted_APE"] = [d["1"]["code"] for d in pred_dicts]
    df["IC"] = [d["IC"] for d in pred_dicts]

    # Example upload – uncomment if you need it
    # upload_parquet(df, "s3://projet-ape/data/compare_model_torch.parquet")

    log.info("Pipeline finished successfully.")


# ──────────────────────────────────────────────────────────────────────────────
# 7️⃣  Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()