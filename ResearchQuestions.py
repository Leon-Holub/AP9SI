import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr, ttest_ind

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


def interpret_correlation(r: float) -> str:
    """Returns qualitative interpretation of correlation strength."""
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "velmi slabá"
    elif abs_r < 0.3:
        strength = "slabá"
    elif abs_r < 0.5:
        strength = "střední"
    else:
        strength = "silná"

    direction = "pozitivní" if r > 0 else "negativní" if r < 0 else "žádná"
    return f"{strength} {direction} korelace"


def analyze_music_frequency_effects(df: pd.DataFrame, save_path: str | None = None, show: bool = True):
    """
    Analyzes whether the frequency of listening to music (hours per day)
    is related to anxiety or depression levels.
    Provides statistical and textual interpretation.
    """
    required_cols = ["Hours per day", "Anxiety", "Depression", "Music effects"]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Missing columns. Required: {required_cols}")
        return

    # --- Čištění dat ---
    data = df.dropna(subset=required_cols).copy()
    data["Hours per day"] = pd.to_numeric(data["Hours per day"], errors="coerce")
    data = data[data["Hours per day"] > 0]

    # --- Korelace ---
    r_anxiety, p_anxiety = pearsonr(data["Hours per day"], data["Anxiety"])
    r_depression, p_depression = pearsonr(data["Hours per day"], data["Depression"])

    print("📈 Korelace mezi počtem hodin poslechu a psychickými ukazateli:")
    print(f"   Anxiety     → r = {r_anxiety:.3f}, p = {p_anxiety:.5f}")
    print(f"   Depression  → r = {r_depression:.3f}, p = {p_depression:.5f}\n")

    # --- Interpretace výsledků ---
    def interpret_result(var_name, r, p):
        text = f"➡️ {var_name}: "
        text += interpret_correlation(r)
        if p < 0.05:
            text += f" (statisticky významná, p = {p:.5f})."
            if r > 0:
                text += " Znamená to, že s rostoucí dobou poslechu se hodnota této proměnné mírně zvyšuje."
            elif r < 0:
                text += " Znamená to, že s rostoucí dobou poslechu tato hodnota spíše klesá."
        else:
            text += f" (nevýznamná, p = {p:.5f})."
            text += " Není prokázána souvislost mezi délkou poslechu a touto proměnnou."
        print(text)

    interpret_result("Anxiety", r_anxiety, p_anxiety)
    interpret_result("Depression", r_depression, p_depression)
    print()

    # --- Grafická část ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.regplot(data=data, x="Hours per day", y="Anxiety", ax=axes[0], color="#2196F3")
    axes[0].set_title("Relationship between Listening Time and Anxiety")
    axes[0].set_xlabel("Hours of Music per Day")
    axes[0].set_ylabel("Anxiety Level")

    sns.regplot(data=data, x="Hours per day", y="Depression", ax=axes[1], color="#E91E63")
    axes[1].set_title("Relationship between Listening Time and Depression")
    axes[1].set_xlabel("Hours of Music per Day")
    axes[1].set_ylabel("Depression Level")

    plt.tight_layout()
    show_or_save_plot(fig, save_path, show)


def analyze_music_while_working(df: pd.DataFrame, save_path: str | None = None, show: bool = True):
    """
    Analyzes whether listening to music while working influences depression levels.

    Parameters:
        df (pd.DataFrame): Dataset with columns ['While working', 'Depression', 'Music effects']
        save_path (str, optional): Path to save the figure.
        show (bool): Whether to display the plot.
    """

    required_cols = ["While working", "Depression", "Music effects"]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Missing required columns: {required_cols}")
        return

    # --- Čištění dat ---
    data = df.dropna(subset=required_cols).copy()
    data = data[data["While working"].isin(["Yes", "No"])]

    # --- Rozdělení podle skupin ---
    group_yes = data[data["While working"] == "Yes"]["Depression"]
    group_no = data[data["While working"] == "No"]["Depression"]

    # --- T-test (porovnání průměrů mezi dvěma skupinami) ---
    t_stat, p_val = ttest_ind(group_yes, group_no, equal_var=False)
    mean_yes = group_yes.mean()
    mean_no = group_no.mean()

    print("🎧 Vliv poslechu hudby při práci na míru deprese:")
    print(f"   Průměrná deprese (poslouchá):     {mean_yes:.2f}")
    print(f"   Průměrná deprese (neposlouchá):   {mean_no:.2f}")
    print(f"   t-stat = {t_stat:.3f}, p-value = {p_val:.5f}")

    if p_val < 0.05:
        print("✅ Rozdíl je statisticky významný – poslech hudby při práci má vliv na úroveň deprese.")
        if mean_yes < mean_no:
            print("   ➡️ Posluchači hudby při práci vykazují nižší míru deprese.")
        else:
            print("   ⚠️ Posluchači hudby při práci vykazují vyšší míru deprese.")
    else:
        print("ℹ️ Nebyl zjištěn statisticky významný rozdíl mezi skupinami.")

    # --- Vizualizace ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=data, x="While working", y="Depression", palette=["#E57373", "#81C784"], ax=ax,
                hue="While working")

    ax.set_title("Depression levels by music listening during work")
    ax.set_xlabel("Listening to music while working")
    ax.set_ylabel("Depression Score")
    plt.tight_layout()

    show_or_save_plot(fig, save_path, show)
