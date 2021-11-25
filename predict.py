# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import tensorflow as tf
import tensorflow as tf
from keras.models import model_from_json
from train import load_images
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plots

# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''


#------------------------------------------------------------------------------
# Euer Code ab hier
    predictions = list()

    #ToDo: Add the same preprocessing steps to predict function as in train function to ensure same data format

    # load previously trained model
    #model = tf.keras.models.load_model('saved_model/my_model')
    ## Check model architecture
    #model.summary()
#
    #predictions = model.predict(ecg_leads)

    # File path
    filepath = 'dataset/saved_model/'
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'AE', 'OE', 'UE', 'SCH', 'One', 'Two', 'Three', 'Four', 'Five']

    # load json and create model
    json_file = open(filepath + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filepath + 'model.h5')
    print("Loaded model from disk")

    # A few random samples
    train_generator, valid_generator, test_generator = load_images()

    # Generate predictions for samples
    predictions = loaded_model.predict(test_generator)  # , num_of_test_samples // batch_size+1)

    num_of_test_samples = test_generator.samples
    batch_size = 32

    # Confution Matrix and Classification Report
    predicted_categories = tf.argmax(predictions, axis=1)
    cm = confusion_matrix(test_generator.classes, predicted_categories)

    print('Confusion Matrix')
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print('Classification Report')
    print(classification_report(test_generator.classes, predicted_categories, target_names=labels))

    # ToDo: Plot ROC curve
    # plots.plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
    # plots.plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
    # plots.plot_roc("Test Baseline", test_generator.classes, predicted_categories, color=plots.colors[0], linestyle='--')
    # plt.legend(loc='lower right')

    #------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
