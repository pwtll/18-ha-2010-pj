# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""
# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from typing import List, Tuple
import preprocess_ecg_lead as prep
import shutil


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

    directory = '../workspace/'

    if '1d' in model_name:
        X_test_tensor = prep.preprocess_ecg_leads(ecg_leads)
    else:
        # generate images of three-r-peak ecg segments from unknown ecg_lead
        for i, (ecg_lead, ecg_name) in enumerate(zip(ecg_leads, ecg_names)):
            # segment ecg lead into segments containing 3 r-peaks each
            ecg_segments = prep.segmentation_ecg_lead(ecg_lead, fs)
            # convert arrays of segmented data into single images and save them in working directory
            test_image_directory = prep.segment_to_single_test_img(ecg_segments, ecg_name, directory)

    # load generated images of 3 r-peak ecg segments
    test_generator = prep.load_test_images(directory)

    # load right model for classification problem
    if is_binary_classifier is True:
        model = prep.load_model_from_name(model_name)

        # tell the model what cost and optimization method to use
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        model = prep.load_model_from_name(model_name)

        # tell the model what cost and optimization method to use
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # if you forget to reset the test_generator you will get outputs in a weird order
    test_generator.reset()

    # Evaluate the model
    # model.evaluate(generator=test_generator, steps=(test_generator.n//test_generator.batch_size))

    # division by the number of images in each subfolder provides one classification for all images
    steps_per_epoch = test_generator.n // test_generator.batch_size          #  test_generator.n // test_generator.batch_size

    if '1d' in model_name:
        pred = model.predict(X_test_tensor)  # history = model.predict(X_test_tensor)
    else:
        # Generate predictions for samples
        pred = model.predict(test_generator, steps=steps_per_epoch, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    labels = {'A': 0, 'N': 1, 'O': 2, '~': 3}
    labels = dict((v, k) for k, v in labels.items())
    predictions_list = [labels[k] for k in predicted_class_indices]
    predictions = list(map(lambda x, y: (x, y), ecg_names, predictions_list))

    print("\n")
    for tuple_ in predictions:
        print("ECG Name: " + tuple_[0] + "\t\t|\tPrediction: " + tuple_[1])

    # delete the created temporary working directory to prevent conflicts during next training
    shutil.rmtree(directory)

    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
