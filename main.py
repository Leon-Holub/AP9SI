from FileLoader import load_and_preprocess_dataset
from PlotCreator import plot_genre_distribution

if __name__ == "__main__":
    df = load_and_preprocess_dataset()
    print(df.head())
    plot_genre_distribution(df)
