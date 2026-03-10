from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

os.makedirs('data/raw', exist_ok=True)

print("Téléchargement du dataset...")
dataset = fetch_ucirepo(id=938)

# Sauvegarde
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)
df.to_csv('data/raw/appendicitis.csv', index=False)
print("Données sauvegardées dans data/raw/appendicitis.csv")