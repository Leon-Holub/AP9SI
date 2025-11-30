import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


def _load_dataset(file_path: str = "mxmh_survey_results.csv") -> pd.DataFrame:
    """
    Loads the 'Music & Mental Health Survey' dataset from Kaggle using KaggleHub.

    Parameters:
        file_path (str): Name of the CSV file within the dataset repository.
                         Default is 'mxmh_survey_results.csv'.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "catherinerasgaitis/mxmh-survey-results",
        file_path
    )

    print("✅ Dataset loaded successfully!")
    print("📊 Shape:", df.shape)
    print("📋 Columns:", df.columns.tolist()[:10], "...")
    print(df.head())
    return df


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataset to include only rows that contain valid (non-empty, non-NaN)
    values in key columns used for research questions.

    Parameters:
        df (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Filtered dataset containing only valid rows.
    """
    required_columns = [
        "Depression",
        "Fav genre",
        "Music effects",
        "Hours per day",
        "Anxiety",
        "Insomnia",
        "While working",
        "Age"
    ]

    # Keep only rows where all required columns are non-empty and not NaN
    filtered_df = df.dropna(subset=required_columns)

    # Remove rows with empty strings or invalid placeholders (like "N/A" or "")
    for col in required_columns:
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().ne("")]
        filtered_df = filtered_df[~filtered_df[col].astype(str).str.contains("N/A|nan", case=False, na=False)]

    print(f"✅ Filtered dataset: {len(filtered_df)} rows remaining out of {len(df)}")
    return filtered_df


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Permissions", "Timestamp"])
    return df


import pandas as pd

def _transform_music_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms 'Music effects' into an ordered categorical variable
    instead of numeric (-1,0,1), as recommended.
    Order: Worsen < No effect < Improve
    """
    if "Music effects" not in df.columns:
        print("⚠️ Column 'Music effects' not found.")
        return df

    df["Music effects"] = pd.Categorical(
        df["Music effects"],
        categories=["Worsen", "No effect", "Improve"],
        ordered=True
    )

    print("🎵 'Music effects' transformed into ordered categorical variable.")
    return df




def _preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_irrelevant_columns(df)
    df = _filter_valid_rows(df)
    df = _transform_music_effects(df)
    print("✅ Dataset preprocessed successfully.")
    return df

def load_and_preprocess_dataset() -> pd.DataFrame:
    df = _load_dataset()
    df_preprocessed = _preprocess_dataset(df)
    return df_preprocessed