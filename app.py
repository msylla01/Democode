from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import base64
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Créer le dossier "uploads" si absent
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Charger le modèle et le préprocesseur
model = joblib.load("subscription_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=file.filename))
    return render_template('index.html')

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    # Calcul des métriques
    report = {
        'rows': data.shape[0],
        'subscription_rate': data['souscrit_offre'].mean() if 'souscrit_offre' in data.columns else None,
        'avg_amount': data['montant'].mean(),
        'avg_frequency': data['frequence_transaction'].mean()
    }

    # Création de toutes les visualisations
    plot_urls = {}
    
    # 1. Transactions par type
    plt.figure(figsize=(10, 6))
    data['type_transaction'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plot_urls['type_transaction'] = save_plot()

    # 2. Répartition par région
    plt.figure(figsize=(10, 6))
    data['region'].value_counts().plot(kind='bar', color='#3498db')
    plt.title("Répartition par région")
    plot_urls['region'] = save_plot()

    # 3. Montants par jour
    plt.figure(figsize=(12, 6))
    data.groupby('date')['montant'].sum().plot(kind='line', color='#e74c3c')
    plt.title("Évolution des montants journaliers")
    plot_urls['daily_amount'] = save_plot()

    # 4. Top 10 clients
    top_clients = data.groupby('id_client')['montant'].sum().nlargest(10)
    plt.figure(figsize=(12, 6))
    top_clients.plot(kind='barh', color='#2ecc71')
    plt.title("Top 10 clients par montant")
    plot_urls['top_clients'] = save_plot()

    # 5. Fréquence par type
    plt.figure(figsize=(10, 6))
    data.groupby('type_transaction')['frequence_transaction'].mean().plot(kind='bar', color='#f1c40f')
    plt.title("Fréquence moyenne par type de transaction")
    plot_urls['freq_by_type'] = save_plot()

    return render_template('analyze.html', report=report, plot_urls=plot_urls, filename=filename)

def save_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()
@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    # Supprimer la colonne 'date' non numérique
    data1 = data.drop(['souscrit_offre', 'date'], axis=1)  # Modification ici
    
# Encoder les colonnes catégorielles
    #preprocessor = ColumnTransformer(
     #   transformers=[
     #       ('cat', OneHotEncoder(), ['type_transaction', 'region'])
      #  ],
       # remainder='passthrough'  # Garder les autres colonnes numériques
    #)

    #data2 = preprocessor.fit_transform(data1)

    # Prétraitement des données
    X = preprocessor.transform(data1)
    predictions = model.predict(X)

    # Ajouter les prédictions au DataFrame
    data['prediction_souscription'] = predictions

    return render_template('predict.html', data=data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
