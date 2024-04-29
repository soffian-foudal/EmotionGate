import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pca

# Carica i dati delle caratteristiche dal file dataset.csv
pca.csvtopca("dataset.csv")
data = pd.read_csv('datasetPca.csv')

# Carica le etichette dal file label.csv
labels = pd.read_csv('../labelling/label.csv')

# Unisci i dati delle caratteristiche con le etichette utilizzando la colonna "Video" come chiave di unione

# Dividi i dati in features (caratteristiche) e target (etichette)
features = data
target = labels['EMOZIONE']

# Definisci i modelli di classificazione
models = [
    ('Logistic Regression', LogisticRegression(max_iter=100000000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC(max_iter=100000000)),
    ('Naive Bayes', GaussianNB())
]

# Crea la sottocartella "modelli" se non esiste
if not os.path.exists("modelli_fullTrain"):
    os.makedirs("modelli_fullTrain")

# Addestra e valuta i modelli utilizzando la cross-validation
for model_name, model in models:

    # Cross-validation
    scores = cross_val_score(model, features, target, cv=10, scoring='accuracy')

    # Stampa i risultati della cross-validation
    print(f"Modello: {model_name}")
    print(f"Accuracy: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")
    print(f"Accuracy Min: {scores.min()}")
    print(f"Accuracy Max: {scores.max()}")
    print()

    # Calcola le metriche per precision, recall e f1-score utilizzando cross_val_predict
    y_pred_cv = cross_val_predict(model, features, target, cv=10)
    precision_cv = precision_score(target, y_pred_cv, average='weighted')
    recall_cv = recall_score(target, y_pred_cv, average='weighted')
    f1_cv = f1_score(target, y_pred_cv, average='weighted')

    print(f"Precision (Cross-Validation): {precision_cv}")
    print(f"Recall (Cross-Validation): {recall_cv}")
    print(f"F1 Score (Cross-Validation): {f1_cv}")
    print()

    print()

    # Addestramento del modello sull'intero dataset
    model.fit(features, target)

    # Salva il modello nella sottocartella "modelli"
    model_filename = f"modelli_fullTrain/{model_name}_model.pkl"
    joblib.dump(model, model_filename)

class_names = model.classes_
print(class_names)
