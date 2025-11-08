from FileLoader import load_and_preprocess_dataset
from PlotCreator import plot_genre_distribution, plot_listening_pie
from ResearchQuestions import analyze_depression_by_genre, analyze_music_frequency_effects, analyze_music_while_working

import pandas as pd


def summarize_columns(df: pd.DataFrame, max_unique: int = 10) -> pd.DataFrame:
    """
    Summarizes all columns in the dataset and infers their data types and value distributions.

    Parameters:
        df (pd.DataFrame): Input dataset.
        max_unique (int): Max number of unique values to display for categorical columns.

    Returns:
        pd.DataFrame: Summary of columns with inferred types and example values.
    """
    summary = []

    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        n_unique = col_data.nunique(dropna=True)

        # --- Heuristika na typ dat ---
        if pd.api.types.is_numeric_dtype(col_data):
            inferred_type = "numeric"
        elif pd.api.types.is_bool_dtype(col_data):
            inferred_type = "boolean"
        elif n_unique < 20:
            inferred_type = "categorical"
        elif pd.api.types.is_string_dtype(col_data):
            inferred_type = "text"
        else:
            inferred_type = "mixed"

        # --- Ukázkové hodnoty ---
        unique_vals = col_data.dropna().unique()[:max_unique]
        example_values = ", ".join(map(str, unique_vals))

        summary.append({
            "Column": col,
            "Inferred Type": inferred_type,
            "Original Dtype": str(dtype),
            "Unique Values": n_unique,
            "Example Values": example_values
        })

    summary_df = pd.DataFrame(summary)
    print("📋 Column summary:\n")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    df = load_and_preprocess_dataset()  # TODO remove people that does not listen to music daily
    print(df.head())
    print(df.columns)
    plot_genre_distribution(df, "plots/genre_distribution.png")
    plot_listening_pie(df, "plots/listening_pie.png")
    summarize_columns(df)

    analyze_depression_by_genre(df, "plots/depression_by_genre.png")
    analyze_music_frequency_effects(df, "plots/music_frequency_effects.png")
    analyze_music_while_working(df, "plots/music_while_working.png")
