import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Charger les données d'entraînement
train_data = pd.read_csv("train_data.csv")

# Supprimer la colonne 'date' non numérique
X_train = train_data.drop(['souscrit_offre', 'date'], axis=1)  # Modification ici
y_train = train_data['souscrit_offre']

# Encoder les colonnes catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['type_transaction', 'region'])
    ],
    remainder='passthrough'  # Garder les autres colonnes numériques
)

X_train_processed = preprocessor.fit_transform(X_train)

# Entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Sauvegarder le modèle et le préprocesseur
joblib.dump(model, "subscription_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("Modèle entraîné avec succès!")
