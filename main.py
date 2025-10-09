from FileLoader import load_and_preprocess_dataset
from PlotCreator import plot_genre_distribution, plot_listening_pie

if __name__ == "__main__":
    df = load_and_preprocess_dataset()  # TODO remove people that does not listen to music daily
    print(df.head())
    plot_genre_distribution(df, "plots/genre_distribution.png")
    plot_listening_pie(df, "plots/listening_pie.png")
