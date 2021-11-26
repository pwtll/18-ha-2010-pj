"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy
import cv2
from joblib import Parallel, delayed
train_path = '../training/'
image_directory = train_path + 'images/'
image_size = 256  # 256


# ToDo: Research: normalization of data necessary?
def main():
    def segmentation(path):
        csv_data = loadmat(path)
        data = np.array(csv_data['val'][0])
        signals = []
        count = 1
        peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=300)[0]
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1] - i)
            x = peaks[count - 1] + diff1 // 2
            y = peaks[count + 1] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        return signals

    def process_singe_img(row):
        filename = row[0]
        label = row[1]
        # Lade MatLab Datei
        ecg_segments = segmentation(os.path.join(train_path, filename + '.mat'))
        signal_to_img(ecg_segments, image_directory, filename, label)
        print(str(row[0]))
    def signal_to_img(array, directory_, filename_, label_):
        if not os.path.exists(directory_ + '/' + label_ + '/' + filename_):
            os.makedirs(directory_ + '/' + label_ + '/' + filename_)

            for count, i in enumerate(array):
                fig = plt.figure(frameon=False)
                plt.plot(i)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                new_filepath = directory_ + '/' + label_ + '/' + filename_ + '/' + '{:05d}'.format(count) + '.png'
                fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
                plt.close(fig)

                # downsampling images to desired image_size
                im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.resize(im_gray, (image_size, image_size))  # , interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
                cv2.imwrite(new_filepath, im_gray)

    with open(os.path.join(train_path, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile, parallele Ausführung
        Parallel(n_jobs=8)(delayed(process_singe_img)(row) for row in csv_reader) #number of cpus here?
        #for row in csv_reader:
        #    filename = row[0]
        #    label = row[1]
        #    # Lade MatLab Datei
        #     ecg_segments = segmentation(os.path.join(train_path, filename + '.mat'))
        #    signal_to_img(ecg_segments, image_directory, filename, label)
        #    print(str(row[0]))




if __name__ == '__main__':
    main()
