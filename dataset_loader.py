import kagglehub

def get_dataset_path():
    path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
    return path
