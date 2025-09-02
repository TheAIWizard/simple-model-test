import mlflow
from mlflow.models import Model
import nltk
import os

nltk.download('stopwords')


def main():
    # model_uri = 'runs:/d12cb827b0d9445dbc134dfc1f37cc7a/pyfunc_model'
    # model_uri = 's3://projet-ape/mlflow-artifacts/32/e5f154b7fd0a4e64811781a17019f81c/artifacts/pyfunc_model'
    
    # The model is logged with an input example
    # Step 1: Set the destination path for the model artifacts
    # model_uri = f"models:/{"test_wrapper_pytorch"}/{"31"}"
    model_uri = 'runs:/1a5496ed174c45459a8fa106ff2a7d87/default'
    dst_path = "../my_model"

    # Step 2: Download/extract the model here *without loading it yet*
    print(dst_path)
    print(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path))

    # Step 3: Append the nltk_data/ folder to nltk path BEFORE loading the model
    nltk_data_path = os.path.join(dst_path, "artifacts", "nltk_data")
    nltk.data.path.append(nltk_data_path)

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(dst_path))
    input_data = pyfunc_model.input_example
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
    print(prediction[0].model_dump())


if __name__ == "__main__":
    main()
