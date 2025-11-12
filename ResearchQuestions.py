import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr, ttest_ind

from PlotCreator import show_or_save_plot

# --- NEW: imports for Q3 (prediction) ---
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os




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

    # Odstraníme NaN hodnoty
    data = df.dropna(subset=required_cols)

    # --- ANOVA test ---
    grouped = [group["Depression"].values for _, group in data.groupby("Fav genre") if len(group) > 2]
    f_stat, p_val = f_oneway(*grouped)
    print(f"📊 ANOVA test:")
    print(f"   F-statistic = {f_stat:.3f}")
    print(f"   p-value     = {p_val:.5f}")
    if p_val < 0.05:
        print("✅ Výsledek je statisticky významný — míra deprese se mezi žánry liší.\n")
    else:
        print("ℹ️ Rozdíly mezi žánry nejsou statisticky významné.\n")

    # --- Přehled průměrných hodnot pro každý žánr ---
    summary = (
        data.groupby("Fav genre")["Depression"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("median", ascending=False)
    )

    print("📋 Přehled hodnot deprese podle žánrů:")
    print(summary.to_string(index=False, formatters={
        "mean": "{:.2f}".format,
        "median": "{:.2f}".format,
        "std": "{:.2f}".format
    }))
    print()

    # --- Vizualizace ---
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

def analyze_disorder_prediction(df: pd.DataFrame, outdir: str = "plots", show: bool = True, threshold: int = 6):
    """
    Q3: Lze na základě hudebních preferencí predikovat riziko duševní poruchy?
    Vytvoří binární cíl 'MentalDisorderRisk' z (Anxiety/Depression/Insomnia/OCD > threshold),
    natrénuje Logistic Regression a Random Forest, uloží metriky a grafy (ROC + feature importance).
    """
    required_targets = ["Anxiety", "Depression", "Insomnia", "OCD"]
    required_features = ["Fav genre", "Hours per day", "While working", "Music effects", "Age"]

    # --- Kontrola sloupců ---
    if not all(c in df.columns for c in required_targets):
        print(f"⚠️ Missing target components: {required_targets}")
        return
    # feature sloupce použijeme jen ty, které v datasetu reálně jsou
    features_present = [c for c in required_features if c in df.columns]
    if not features_present:
        print("⚠️ No predictive features found.")
        return

    # vytvoří podsložku pro tuto analýzu (např. plots/Q3_predikce)
    outdir = os.path.join(outdir, "Q3_predikce")
    os.makedirs(outdir, exist_ok=True)

    # --- Target: MentalDisorderRisk ---
    df = df.copy()
    for num_col in ["Anxiety", "Depression", "Insomnia", "OCD", "Hours per day", "Age"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["MentalDisorderRisk"] = (
        (df[required_targets] > threshold).any(axis=1)
    ).astype(int)

    # --- Příprava X,y ---
    model_df = df[features_present + ["MentalDisorderRisk"]].dropna().copy()
    # one-hot pro kategoriální
    X = pd.get_dummies(model_df[features_present], drop_first=True)
    y = model_df["MentalDisorderRisk"].astype(int)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # --- Modely ---
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    # --- Metriky ---
    metrics = pd.DataFrame([
        {"model": "RandomForest",
         "accuracy": accuracy_score(y_test, rf_pred),
         "f1": f1_score(y_test, rf_pred),
         "roc_auc": roc_auc_score(y_test, rf_prob)},
        {"model": "LogisticRegression",
         "accuracy": accuracy_score(y_test, lr_pred),
         "f1": f1_score(y_test, lr_pred),
         "roc_auc": roc_auc_score(y_test, lr_prob)},
    ])
    print("📊 Q3 metrics:\n", metrics.round(4).to_string(index=False))

    # --- ROC křivky ---
    fig1 = plt.figure()
    RocCurveDisplay.from_estimator(rf, X_test, y_test)
    plt.title("ROC – RandomForest (Q3)")
    show_or_save_plot(fig1, os.path.join(outdir, "roc_randomforest_q3.png"), show)

    fig2 = plt.figure()
    RocCurveDisplay.from_estimator(lr, X_test, y_test)
    plt.title("ROC – LogisticRegression (Q3)")
    show_or_save_plot(fig2, os.path.join(outdir, "roc_logreg_q3.png"), show)

    # --- Feature importance (RF) ---
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    topk = importances.head(15)[::-1]
    fig3, ax = plt.subplots(figsize=(7, 6))
    topk.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 feature importances – RF (Q3)")
    ax.set_xlabel("Gini importance")
    plt.tight_layout()
    show_or_save_plot(fig3, os.path.join(outdir, "feature_importance_rf_q3.png"), show)

    # --- Ulož metriky do CSV (pro tabulky v práci) ---
    metrics.to_csv(os.path.join(outdir, "metrics_q3.csv"), index=False)

    return metrics

