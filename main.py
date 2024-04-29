import os
import sys
from effettuaPrevisioni import effettua_previsioni
import subprocess
from modelTraining.preprocessing.segmentazione import rimuovi_ripetizioni

def esegui_file_python(path_file):
    subprocess.run(['python', path_file])

def chiamare_segmentazione(percorso):
    subprocess.call(["python", "segmentazioneProb.py", percorso])

def main():




    # Verifica che siano stati passati due argomenti da linea di comando
    if len(sys.argv) != 4:
        print("Usage: python main.py <cartella_video> <file_xml>")
        return

    cartella_video_prof = sys.argv[1]

    file_xml_prof = sys.argv[2]
    modelNumber = int(sys.argv[3])
    print("check:" + cartella_video_prof)
    # Sostituisci i caratteri "\" con "/"
    cartella_video_prof = cartella_video_prof.replace("\\", "/")
    print("converted to:" + cartella_video_prof)


    print("check:" + file_xml_prof)
    file_xml_prof = file_xml_prof.replace("\\", "/")
    print("converted to:" + file_xml_prof)
    print("check: Model Number: " + str(modelNumber))


    cartella_video_prof = os.path.join(cartella_video_prof, "videos_segmenti")

    print("calling... effettua previsioni....")
    effettua_previsioni(cartella_video_prof,file_xml_prof, modelNumber)


if __name__ == "__main__":
    main()
