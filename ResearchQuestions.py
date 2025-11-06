import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

from PlotCreator import show_or_save_plot


def analyze_depression_by_genre(df: pd.DataFrame, save_path: str | None = None, show: bool = True):
    """
    Analyzes whether depression levels differ across music genres.
    Performs ANOVA test and creates a boxplot.

    Parameters:
        df (pd.DataFrame): Dataset containing 'Depression' and 'Fav genre'.
        save_path (str, optional): Path to save the plot.
        show (bool): Whether to display the plot.
    """

    # Kontrola sloupců
    required_cols = ["Depression", "Fav genre"]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Missing required columns: {required_cols}")
        return

    data = df.dropna(subset=required_cols)

    grouped = [group["Depression"].values for _, group in data.groupby("Fav genre") if len(group) > 2]
    f_stat, p_val = f_oneway(*grouped)
    print(f"📊 ANOVA test:")
    print(f"   F-statistic = {f_stat:.3f}")
    print(f"   p-value     = {p_val:.5f}")
    if p_val < 0.05:
        print("✅ Výsledek je statisticky významný — míra deprese se mezi žánry liší.")
    else:
        print("ℹ️ Rozdíly mezi žánry nejsou statisticky významné.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x="Fav genre", y="Depression", ax=ax, palette="Set2", hue="Fav genre")

    ax.set_title("Depression levels across favorite music genres", fontsize=14)
    ax.set_xlabel("Favorite Music Genre", fontsize=12)
    ax.set_ylabel("Depression Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    show_or_save_plot(fig, save_path, show)
