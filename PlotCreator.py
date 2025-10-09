import matplotlib.pyplot as plt
import pandas as pd

def plot_genre_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Creates a bar chart showing the frequency of favorite music genres.

    Parameters:
        df (pd.DataFrame): The dataset containing 'Fav genre' column.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    if "Fav genre" not in df.columns:
        print("⚠️ Column 'Fav genre' not found in dataset.")
        return

    genre_counts = df["Fav genre"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts.plot(kind="bar", edgecolor="black", ax=ax)

    ax.set_title("Frequency of Favorite Music Genres", fontsize=14)
    ax.set_xlabel("Music Genre", fontsize=12)
    ax.set_ylabel("Number of Respondents", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    show_or_save_plot(fig, save_path)

def show_or_save_plot(fig: plt.Figure, save_path: str | None = None, show: bool = True) -> None:
    """
    Displays or saves a matplotlib figure depending on whether save_path is provided.

    Parameters:
        fig (plt.Figure): The matplotlib figure to display or save.
        save_path (str, optional): If provided, the figure is saved to this path.
        show (bool): Whether to display the plot interactively (default True).
    """
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")

    if show and not save_path:
        plt.show()

    plt.close(fig)