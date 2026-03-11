import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_data():
    # télécharger les données depuis UCI Repository
    dataset = fetch_ucirepo(id=938)
    X = dataset.data.features
    y = dataset.data.targets["Diagnosis"]
    return X, y
def clean_data(X, y):
    # Remplir les cases vides avec la valeur médiane
    X = X.fillna(X.median(numeric_only=True))
    
    # Convertir les colonnes texte en chiffres
    X = pd.get_dummies(X)
    
    #"Convertir y : ""appendicitis" = 1," "no appendicitis" =0" 
    y = (y == "appendicitis").astype(int)
    
    return X, y
def split_data(X, y):
    # couper 80% / 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test
def optimize_memory(df):
    # réduire la mémoire
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
def normalize_data(X_train, X_test):
    "mettre à la même échelle"
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, scaler
if __name__ == "__main__":
    X, y = load_data()
    X, y = clean_data(X, y)
    X = optimize_memory(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = normalize_data(X_train, X_test)
    print(" Données prêtes !")
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")