"""
Data processing module for appendicitis dataset.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Colonnes de fuite de données — connues APRÈS le diagnostic, pas au triage
LEAKAGE_COLS = [
    'Management',           # décision thérapeutique → après diagnostic
    'Severity',             # sévérité classifiée → après diagnostic
    'Length_of_Stay',       # durée hospitalisation → après diagnostic
    'Perforation',          # constatée en chirurgie
    'Appendicular_Abscess', # constatée en chirurgie
]


def load_data(filepath="data/raw/appendicitis.csv"):
    """
    Load the appendicitis dataset from CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame: Loaded data
    """
    df = pd.read_csv(filepath)
    return df


def optimize_memory(df):
    """
    Optimise l'utilisation mémoire en convertissant les types de données.
    float64 → float32, int64 → int32

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame optimisé
    """
    before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")

    after = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Mémoire avant : {before:.2f} MB")
    print(f"Mémoire après : {after:.2f} MB")
    print(f"Gain : {before - after:.2f} MB")

    return df


def preprocess_data(df, target_col="Diagnosis", test_size=0.2, random_state=42):
    """
    Preprocess the dataset:
    - Handle missing target values
    - Remove leakage columns (known only after diagnosis)
    - Encode categorical variables
    - Impute missing values
    - Optimize memory
    - Split into train/test sets

    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test, le_target)
    """
    # Supprimer les lignes où la cible est manquante
    df = df.dropna(subset=[target_col])

    # Encoder la cible
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))

    # Supprimer la cible ET les colonnes de fuite
    cols_to_drop = [target_col] + [c for c in LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    removed = [c for c in LEAKAGE_COLS if c in df.columns]
    print(f"Colonnes de fuite supprimées : {removed}")
    print(f"Features restantes : {X.shape[1]}")

    # Encoder les variables catégorielles
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = X[col].fillna("unknown")
        X[col] = le.fit_transform(X[col].astype(str))

    # Imputer les valeurs manquantes avec la médiane
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print(f"Valeurs manquantes après imputation: {X.isnull().sum().sum()}")
    print(f"Taille du dataset: {X.shape}")
    print(f"Classes: {le_target.classes_}")

    # Optimiser la mémoire
    X = optimize_memory(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, le_target


def get_feature_names(X):
    """
    Get feature names from DataFrame.

    Args:
        X: Feature DataFrame

    Returns:
        list: List of feature names
    """
    return X.columns.tolist()


def get_data_summary(df):
    """
    Get summary statistics of the dataset.

    Args:
        df: Input DataFrame

    Returns:
        dict: Summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object"]
        ).columns.tolist(),
    }
    return summary


if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")

    summary = get_data_summary(df)
    print(f"\nNumeric columns: {summary['numeric_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")

    X_train, X_test, y_train, y_test, le_target = preprocess_data(df)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")