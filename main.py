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
    # model_uri = 's3://projet-ape/mlflow-artifacts/32/e5f154b7fd0a4e64811781a17019f81c/artifacts/pyfunc_model'
    # model_uri = 'runs:/fbcd5c2f97e645f1850dbfc3f139c564/default'
    # model_uri = 'runs:/1b6616da89eb45cea458012c5cb6820a/default'
    model_uri = f"models:/{"FastText-APE-nace2025"}/{"9"}"
    dst_path = "../my_model"

    # Step 2: Download/extract the model here *without loading it yet*
    print(dst_path)
    print(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path))

    # Step 3: Append the nltk_data/ folder to nltk path BEFORE loading the model
    nltk_data_path = os.path.join(dst_path, "artifacts", "nltk_data")
    nltk.data.path.append(nltk_data_path)

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(dst_path, "default"))
    # input_data = pyfunc_model.input_example

    libelle = ["apporteur d'affaires",
               "conseiller mandataire vdi au sein de vorwerk france",
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

    print(prediction)
    print(prediction[0])

    # Lire le CSV dans un DataFrame
    df = pd.read_csv('resultats_comparaison_ape_complet.csv', encoding='utf8', delimiter=';')
    print(df)

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

    predictions = pyfunc_model.predict(
        input_data
    )

    print(text_input)

    predicted_labels, predicted_probs = predictions

    df = df[["Texte_Descriptif", "APE_Propose", "APE_DV2"]]
    # 4. Nettoyer les résultats et les ajouter au DataFrame
    df['Predicted_APE'] = [label[0].replace('__label__', '') for label in predicted_labels]
    df['Prediction_Probability'] = [prob[0] for prob in predicted_probs]

    # 5. Afficher le DataFrame résultant
    print(df)

    # upload_parquet(df, 's3://projet-ape/data/compare_model_stock.parquet')


if __name__ == "__main__":
    main()
