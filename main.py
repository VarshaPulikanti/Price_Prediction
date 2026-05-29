from src.data import download_real_estate_dataset
from src.train import train_model

if __name__ == "__main__":
    download_real_estate_dataset()
    results = train_model()
    print("Training complete.")
    print(results)
