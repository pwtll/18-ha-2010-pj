# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""

# import socket
# def guard(*args, **kwargs):
#     raise Exception("Internet Access Forbidden")
# socket.socket = guard

from predict import predict_labels
from wettbewerb import load_references, save_predictions
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test_pred/')  # ToDo: reset to: default='../test/'
    parser.add_argument('--model_name', action='store',type=str,default='dataset/saved_model/custom_model_2d_cnn_v3_four_classes-num_epochs_10-batch_size_32-image_size_128_20211129-154826')  # ,default='model.npy')
    args = parser.parse_args()
    
    ecg_leads,ecg_labels,fs,ecg_names = load_references(args.test_dir)  # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
    
    start_time = time.time()
    predictions = predict_labels(ecg_leads,fs,ecg_names,model_name=args.model_name)
    pred_time = time.time()-start_time
    
    save_predictions(predictions) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")
