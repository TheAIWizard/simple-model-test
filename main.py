from mlflow.models import Model
import s3fs
import mlflow
import nltk
import os
import pandas as pd

nltk.download('stopwords')


def get_filesystem():
    """
    Configure and return a S3-compatible filesystem (MinIO / AWS S3).
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def upload_parquet(df: pd.DataFrame, path: str):
    """
    Save DataFrame as Parquet to local or S3.
    """
    fs = get_filesystem()
    return df.to_parquet(path, index=False, filesystem=fs)


def main():
    # The model is logged with an input example
    # Step 1: Set the destination path for the model artifacts
    # model_uri = 's3://projet-ape/mlflow-artifacts/31/f93f3a6efbb649bca00cb4e5aecc298a/artifacts/pyfunc_model'
    # model_uri = 'runs:/fbcd5c2f97e645f1850dbfc3f139c564/default'
    # model_uri = 'runs:/1b6616da89eb45cea458012c5cb6820a/default'
    # model_uri = f"models:/{"FastText-pytorch-2025"}/{"8"}"
    model_uri = 'runs:/05639a37f98244eea3c06cdeeecd9631/pyfunc_model'
    dst_path = "../my_model"

    # Step 2: Download/extract the model here *without loading it yet*
    print(dst_path)
    print(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path))

    # Step 3: Append the nltk_data/ folder to nltk path BEFORE loading the model
    nltk_data_path = os.path.join(dst_path, "artifacts", "nltk_data")
    nltk.data.path.append(nltk_data_path)

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(dst_path, "pyfunc_model"))

    libelle = ["vente a distance",
               "vente à distance sur catalogue",
               "apporteur d'affaire digital",
               "ACHAT REVENTE SUR INTERNET HABILLEMENT ACCESSOIRES ET CHAUSSURES",
               "Achat/vente de vinyles d'occasion en ligne",
               "PENSION POUR ANIMAUX DE COMPAGNIE",
               "Nettoyage et entretien extérieur ( murs, terrasse, sols exterieurs, clôture, toiture, gouttière, espace vert )  traitement anti mousse et ap",
               "création de site internet sans programmation (design, ergonomie...)",
               "elevage bovin",
               "Graphiste, conception de supports",
               "coiffure hors salon",
               "coach sportif",
               "PROFESSEUR DE NATATION",
               "coach en entreprise",
               "Service de coaching, conseil sportifs et nutritionnels, individuel ou collectif, vente de programmes sportifs et alimentaires personnalisés"]

    input_data = pd.DataFrame({
        "libelle": libelle,
        "CJ": [None] * 15,  # Or pd.NA, or np.nan if you import numpy
        "SRF": [None] * 15,
        "NAT": [None] * 15,
        "TYP": ["X"] * 15,
        "CRT": [None] * 15,
        "activ_sec_agri_et": [None] * 15,
        "activ_nat_lib_et": [None] * 15
    })

    # input_data = pyfunc_model.input_example
    input_data = []
    for text in libelle:
        # Création du dictionnaire d'entrée avec SEULEMENT le champ essentiel
        input_item = {
            "description_activity": text,
            }
        input_data.append(input_item)
    print(pyfunc_model)
    print(input_data)
    print("MODEL_ID")
    print(pyfunc_model._model_id)
    print("RUN_ID")
    print(pyfunc_model.metadata.run_id)
    print(pyfunc_model.metadata)
    print(pyfunc_model.model_config)

    prediction = pyfunc_model.predict(
        input_data
    )

    golden_tests = pd.read_csv('golden_tests.csv', encoding='utf8', delimiter=';')
    upload_parquet(golden_tests, 's3://projet-ape/data/08112022_27102024/naf2025/golden_tests.parquet')

    golden_tests["description_activity"]=golden_tests["libelle"]
    golden_tests["activity_permanence_status"]=golden_tests["CRT"].fillna("NaN")

    predictions_payload = []
    list_of_dicts = golden_tests[["description_activity", "activity_permanence_status"]].to_dict(orient='records')
    for record in list_of_dicts:
        cleaned_record = {k: v for k, v in record.items() if v != "NaN"}
        predictions_payload.append(cleaned_record)
    print(predictions_payload)
    print(list(predictions_payload))
    predictions = pyfunc_model.predict(
        predictions_payload
    )
    pred_dump = [prediction.model_dump() for prediction in predictions]
    golden_tests["APE_prediction"] = [pred["1"]["code"] for pred in pred_dump]
    golden_tests["IC"] = [pred["IC"] for pred in pred_dump]

    print(golden_tests)
    concordance_mask = (golden_tests['nace2025'] == golden_tests['APE_prediction'])
    taux_concordance = concordance_mask.mean()
    print(taux_concordance)
    print(golden_tests["IC"].mean())
    print(golden_tests["IC"].median())
    gt_non_concordants = golden_tests[~concordance_mask]
    print(gt_non_concordants[["libelle", "APE_prediction", "IC", "CRT"]])
    upload_parquet(golden_tests, 's3://projet-ape/data/golden_tests_results.parquet')
    upload_parquet(gt_non_concordants, 's3://projet-ape/data/golden_tests_error.parquet')

    # Lire le CSV dans un DataFrame
    df = pd.read_csv('resultats_comparaison_ape.csv', encoding='utf8', delimiter=' ')
   

    # 3. Extraire la colonne 'Texte_Descriptif' et appliquer la prédiction
    text_input = df['Texte_Descriptif'].tolist()

    input_data = pd.DataFrame({
        "libelle": text_input,
        "CJ": [None] * len(text_input),  # Or pd.NA, or np.nan if you import numpy
        "SRF": [None] * len(text_input),
        "NAT": [None] * len(text_input),
        "TYP": [None] * len(text_input),
        "CRT": [None] * len(text_input),
        "activ_sec_agri_et": [None] * len(text_input),
        "activ_nat_lib_et": [None] * len(text_input)
        })

    pytorch_input_data = []
    for text in text_input:
        # Création du dictionnaire d'entrée avec SEULEMENT le champ essentiel
        input_item = {
            "description_activity": text,
            }
        pytorch_input_data.append(input_item)

    predictions = pyfunc_model.predict(
        pytorch_input_data
    )

    # predicted_labels, predicted_probs = predictions
    pred_dump = [prediction.model_dump() for prediction in predictions]
    predicted_labels = [pred["1"]["code"] for pred in pred_dump]
    predicted_IC = [pred["IC"] for pred in pred_dump]

    df = df[["Texte_Descriptif", "APE_Propose", "APE_DV2"]]
    # 4. Nettoyer les résultats et les ajouter au DataFrame
    # df['Predicted_APE'] = [label[0].replace('__label__', '') for label in predicted_labels]
    # df['Prediction_Probability'] = [prob[0] for prob in predicted_probs]
    df['Predicted_APE'] = predicted_labels
    df['IC'] = predicted_IC

    # 5. Afficher le DataFrame résultant
    # print(df)

    # upload_parquet(df, 's3://projet-ape/data/compare_model_torch.parquet')


if __name__ == "__main__":
    main()
