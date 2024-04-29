import os

import joblib
import pandas as pd
from sklearn.decomposition import PCA


def riduci_dimensioni_con_pca(nuovo_csv, pca_model_path):

    pca_model = joblib.load(pca_model_path)
    # Leggi il nuovo file CSV utilizzando pandas
    dataframe_nuovo = pd.read_csv(nuovo_csv)

    # Esegui la standardizzazione dei nuovi dati
    nuovo_standardizzato = (dataframe_nuovo - dataframe_nuovo.mean()) / dataframe_nuovo.std()

    # Applica la PCA ai nuovi dati standardizzati
    dati_ridotti_nuovi = pca_model.transform(nuovo_standardizzato)

    # Crea un nuovo dataframe con i dati ridotti
    dati_ridotti_nuovi_dataframe = pd.DataFrame(dati_ridotti_nuovi)

    # Salva i dati ridotti su un file CSV
    output_file_name = "datasetPca.csv"
    folder_path = os.path.dirname(nuovo_csv)
    output_file_path = os.path.join(folder_path, output_file_name)
    dati_ridotti_nuovi_dataframe.to_csv(output_file_path, index=False)

    return output_file_path


def csvtopca(og_csv):
    # Leggi il file CSV utilizzando pandas
    dataframe = pd.read_csv(og_csv)

    # Esegui la standardizzazione dei dati
    data_standardizzato = (dataframe - dataframe.mean()) / dataframe.std()

    # Crea un'istanza di PCA specificando il numero di componenti desiderate
    pca = PCA(n_components=61)

    # Applica la PCA ai dati standardizzati
    dati_ridotti = pca.fit_transform(data_standardizzato)

    # Salva il modello PCA allenato utilizzando joblib
    model_filename = "pcatrained.pkl"

    joblib.dump(pca, model_filename)

    # Crea un nuovo dataframe con i dati ridotti
    dati_ridotti_dataframe = pd.DataFrame(dati_ridotti)

    # Salva i dati ridotti su un file CSV
    output_file_name = "datasetPca.csv"
    output_file_path = os.path.join(os.path.dirname(og_csv), output_file_name)
    dati_ridotti_dataframe.to_csv(output_file_path, index=False)

    print("Dati ridotti salvati su file CSV:", output_file_path)
    print("Modello PCA allenato salvato su:", model_filename)