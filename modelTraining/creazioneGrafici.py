import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pca

# Carica i dati delle caratteristiche dal file dataset.csv
pca.csvtopca("dataset.csv")
data = pd.read_csv('dataset.csv')

# Carica le etichette dal file label.csv
labels = pd.read_csv('../labelling/label.csv')

# Unisci i dati delle caratteristiche con le etichette utilizzando la colonna "Video" come chiave di unione


# Dividi i dati in features (caratteristiche) e target (etichette)
features = data
target = labels['EMOZIONE']

# Dividi i dati in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# Definisci i modelli di classificazione
models = [
    ('Logistic Regression', LogisticRegression(max_iter=100000000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC(max_iter=100000000)),
    ('Naive Bayes', GaussianNB())
]

# Liste per memorizzare i risultati dell'accuratezza dei modelli
model_names = []
accuracy_means = []
accuracy_stds = []

# Addestra e valuta i modelli utilizzando la cross-validation
for model_name, model in models:

    # Cross-validation
    scores = cross_val_score(model, features, target, cv=10, scoring='accuracy')

    # Calcola la media e la deviazione standard dell'accuratezza
    accuracy_mean = scores.mean()
    accuracy_std = scores.std()

    # Memorizza i risultati dell'accuratezza
    model_names.append(model_name)
    accuracy_means.append(accuracy_mean)
    accuracy_stds.append(accuracy_std)

    # Stampa i risultati della cross-validation
    print(f"Modello: {model_name}")
    print(f"Accuracy: {accuracy_mean}")
    print(f"Standard Deviation: {accuracy_std}")
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

    # Addestramento del modello sul set di addestramento completo
    model.fit(X_train, y_train)

    # Previsioni sul set di test
    y_pred = model.predict(X_test)

    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Modello: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()
    # Crea un array di valori per le fold
    fold_numbers = np.arange(1, len(scores) + 1)

    # Visualizza il grafico con l'andamento dell'accuratezza per ogni fold
    plt.plot(fold_numbers, scores, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuratezza')
    plt.title('Andamento dell\'accuratezza per ogni fold MODELLO: ' + model_name )
    plt.xticks(fold_numbers)
    plt.show()

# Visualizza il grafico con i risultati dell'accuratezza dei modelli
plt.figure(figsize=(10, 6))
plt.errorbar(model_names, accuracy_means, yerr=accuracy_stds, fmt='o')
plt.xlabel('Classificatori')
plt.ylabel('Accuratezza')
plt.title('Pipeline dei risultati della cross-validation ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
