from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Download your dataset
api.dataset_download_files(
    'VarshiniN2006/sepsis-multimodel-dataset',
    path='data/',
    unzip=True
)

print("Dataset downloaded successfully ✔")