from FileLoader import load_and_preprocess_dataset

if __name__ == "__main__":
    df = load_and_preprocess_dataset()
    print(df.head())
