import pandas as pd
import numpy as np

# Générer 8000 entrées
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=8000, freq='D'),
    'montant': np.random.randint(100, 10000, 8000),
    'type_transaction': np.random.choice(['dépôt', 'retrait', 'paiement'], 8000),
    'id_client': np.random.randint(1, 200, 8000),
    'region': np.random.choice(['Abidjan', 'Bouaké', 'San-Pédro', 'Yamoussoukro'], 8000),
    'frequence_transaction': np.random.randint(1, 30, 8000),
    'souscrit_offre': np.random.choice([0, 1], 8000, p=[0.7, 0.3])  # 30% souscrivent
})

# Diviser en entraînement (70%) et test (30%)
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)

# Sauvegarder
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("uploads/test_data.csv", index=False)

print("Données générées et sauvegardées !")
