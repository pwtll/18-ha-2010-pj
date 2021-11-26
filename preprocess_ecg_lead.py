import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy
import cv2
from keras.models import model_from_json
import glob

image_size = 512  # 256
sampling_rate = 300


def load_latest_model():
    # File path containing saved models
    filepath = 'dataset/saved_model/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(filepath + '*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getmtime)
    head, tail = os.path.split(latest_file)
    model_name = tail.split('.')[0]

    # load json and create model
    json_file = open(filepath + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filepath + model_name + '.h5')
    print("Loaded model from disk")

    return loaded_model


def load_model_from_name(model_name):
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    print("Loaded model from disk")

    return loaded_model


def segmentation(path_):
    csv_data = loadmat(path_)
    data = np.array(csv_data['val'][0])
    signals = []
    count = 2
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=sampling_rate)[0]
    for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
        diff1 = abs(peaks[count - 2] - i)
        diff2 = abs(peaks[count + 2] - h)
        x = peaks[count - 2] + diff1 // 2
        y = peaks[count + 2] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 3
    return signals

def segment_to_img(array_, directory_, filename_, label_):
    new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

    for count, i in enumerate(array_):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
        fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # downsampling images to desired image_size
        im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (image_size, image_size)) #, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
        cv2.imwrite(new_filepath, im_gray)

    return new_file_directory

def segment_ecg_lead(ecg_leads, fs):
    #csv_data = loadmat(path_)
    data = ecg_leads  #np.array(csv_data['val'][0])
    signals = []
    count = 2
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=fs)[0]
    for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
        diff1 = abs(peaks[count - 2] - i)
        diff2 = abs(peaks[count + 2] - h)
        x = peaks[count - 2] + diff1 // 2
        y = peaks[count + 2] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 3
    return signals

def segment_to_test_img(array_, ecg_name_, directory_):
    new_file_directory = directory_ + '/' + ecg_name_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

    for count, i in enumerate(array_):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
        fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # downsampling images to desired image_size
        im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (image_size, image_size))  # , interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
        cv2.imwrite(new_filepath, im_gray)

    return directory_