# Utilizzo dello script di previsione
import joblib
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os

import null
import pandas as pd
from sklearn.metrics import accuracy_score

from modelTraining import pca
from support import feature_csv,label_csv

# Carica il modello preaddestrato
modello = None

def caricaModello(modelNumber):
    if modelNumber == 1:
        model_path = "modelTraining/modelli/Logistic Regression_model.pkl"
    elif modelNumber == 2:
        model_path = "modelTraining/modelli/Decision Tree_model.pkl"
    elif modelNumber == 3:
        model_path = "modelTraining/modelli/Random Forest_model.pkl"
    else:
        raise ValueError("Il parametro modelNumber deve essere 1, 2 o 3.")

    model = joblib.load(model_path)
    return model

# Funzione per fare previsioni sui video utilizzando il modell

def effettua_previsioni(cartella_video, file_xlsx, modelnumber):
    nomi_modelli = ["", "Logistic Regression", "Decision Tree", "Random Forest"]

    modello = caricaModello(modelnumber)
    print("DONE, INSIDE EFFETTUA PREVISIONI")
    print("chiamata feature con path"+ cartella_video)
    #feature_csv.crea_csv(cartella_video)
    print("DONE, creazione dataset test")
    #label_csv.crea_csv(file_xlsx)
    print("DONE, creazione label test")

    pca.riduci_dimensioni_con_pca("support/csv/dataset.csv","modelTraining/pcatrained.pkl")
    # Prepara un DataFrame per le caratteristiche
    df_caratteristiche = pd.read_csv('support/csv/datasetPca.csv')

    # Prepara un DataFrame per le etichette
    df_etichette = pd.read_csv('support/csv/label.csv')

    # Combina i DataFrame di caratteristiche ed etichette
    #df_completo = pd.merge(df_caratteristiche, df_etichette, on='Video')



    features = df_caratteristiche
    # Effettua previsioni utilizzando il modello
    previsioni = modello.predict(features)

    # Estrai la colonna 'EMOZIONE' che contiene le labels
    labels = df_etichette['EMOZIONE'].tolist()
    target = df_etichette['EMOZIONE']
    # Stampa le labels
    print("Labels:", labels)
    # Stampa le previsioni
    print(previsioni)
    print("Modello in uso:", nomi_modelli[modelnumber])

    accuracy = accuracy_score(labels, previsioni)
    print("Accuracy:", accuracy)
    f1_cv = f1_score(labels, previsioni, average='weighted')
    print("f1 score:", f1_cv)
    precision_cv = precision_score(labels, previsioni, average='weighted')
    print("precision:", precision_cv)
    recall_cv = recall_score(labels, previsioni, average='weighted')
    print("recall:", recall_cv)
    matrice_confusione = confusion_matrix(labels, previsioni)
    print("Matrice di confusione:")
    print(matrice_confusione)
    report_class = classification_report(labels, previsioni)
    print("classification report", report_class)




#effettua_previsioni( "C:/Users/xsoff/Desktop/biometria/TEST/test", "C:/Users/xsoff/Desktop/biometria/TEST/responses_test.xlsx")
