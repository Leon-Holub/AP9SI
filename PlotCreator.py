import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

sns.set_theme(
    style="whitegrid",
    palette="Set2",
    font_scale=1.2,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["legend.frameon"] = False

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


def plot_listening_pie(df: pd.DataFrame, save_path: str | None = None, show: bool = True) -> None:
    """
    Creates a pie chart showing how many respondents listen to music daily (yes/no),
    including a legend with absolute counts.

    Parameters:
        df (pd.DataFrame): Dataset containing 'Hours per day' column.
        save_path (str, optional): If provided, saves the plot to this path.
        show (bool): Whether to display the plot (default True).
    """
    if "Hours per day" not in df.columns:
        print("⚠️ Column 'Hours per day' not found in dataset.")
        return

    df_copy = df.copy()
    df_copy["Hours per day"] = pd.to_numeric(df_copy["Hours per day"], errors="coerce")

    df_copy["Listens music daily"] = df_copy["Hours per day"].apply(
        lambda x: "Yes" if pd.notna(x) and x > 0 else "No"
    )

    counts = df_copy["Listens music daily"].value_counts()
    labels = counts.index
    values = counts.values

    colors = ["#4CAF50", "#E57373"]  # zelená = ano, červená = ne

    # Vykreslení koláče
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "white", "fontsize": 11}
    )

    legend_labels = [f"{label}: {count} respondents" for label, count in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        title="Listening daily",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10
    )

    ax.set_title("Do people listen to music daily?", fontsize=14)
    plt.tight_layout()

    show_or_save_plot(fig, save_path, show)

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

    if show:
        plt.show()

    plt.close(fig)
