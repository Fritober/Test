import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from flask import Flask, request, jsonify

# Chargement des données Breast Cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialisation et entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
accuracy = accuracy_score(y_test, model.predict(X_test))
report = classification_report(y_test, model.predict(X_test), output_dict=True)
conf_matrix = confusion_matrix(y_test, model.predict(X_test))

# Sauvegarde du modèle
joblib.dump(model, 'random_forest_model.pkl')
print("Modèle sauvegardé sous 'random_forest_model.pkl'")

# Création de l'API Flask
app = Flask(__name__)

# Endpoint pour prédire
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint pour obtenir la performance du modèle
@app.route('/performance', methods=['GET'])
def performance():
    return jsonify({'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': conf_matrix.tolist()})

if __name__ == '__main__':
    app.run(debug=True)