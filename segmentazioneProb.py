import sys
from modelTraining.preprocessing.segmentazione import rimuovi_ripetizioni

# Ottieni il percorso dalla linea di comando
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print("Percorso non specificato.")
    exit()

# Esegui la funzione rimuovi_ripetizioni sul percorso specificato
rimuovi_ripetizioni(path)
