# ToDo: Update requirements.txt at the end of project
# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from glob import glob
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plots
import models
import preprocess_ecg_lead as prep
from wettbewerb import load_references


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

epochs = 10
batch_size = 64
image_size = 256
IMAGE_SIZE = [image_size, image_size]               # re-size all the images to this
binary_classification = False
save_trained_model = True
class_indices = {'A': 0, 'N': 1, 'O': 2, '~': 3}

if binary_classification:
    train_path = '../training/images_hamilton_' + str(image_size) + '_2_classes/'        # Enter the directory of the training images seperated in 2 classes
else:
    train_path = '../training/images_hamilton_' + str(image_size) + '/'                  # Enter the directory of the training images seperated in 4 classes
chkp_filepath = 'dataset/model_training_checkpoints'                             # Enter the filename you want your model to be saved as


def get_num_of_classes():
    return len(glob(train_path + '/*'))


# load image data and convert it to the right dimensions to train the model
def load_training_images():
    if binary_classification:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'

    # ToDo: wähle aus den erstellten Bildern das auffälligste aus (zb mit niedrigstem/höchsten Frequenzpeak in fft)
    train_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # , label_mode='categorical')  # rescale=1./255 to scale colors to values between [0,1]
    train_generator = train_gen.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode='rgb',
                                                    shuffle=False,
                                                    batch_size=batch_size,
                                                    subset='training',
                                                    class_mode=class_mode)
                                                    # preprocessing_function=csv_to_three_peaks_image) # ToDO: define automatic preprocessing_function when implementation is finished

    valid_generator = train_gen.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode='rgb',
                                                    shuffle=False,
                                                    batch_size=batch_size,
                                                    subset='validation',
                                                    class_mode=class_mode)

    class_occurences = dict(zip(*np.unique(train_generator.classes, return_counts=True)))
    print("class_occurences: \t" + str(class_occurences))

    return train_generator, valid_generator


# load image data and convert it to the right dimensions to test the model on unseen data
def load_test_images(valid_path):
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, color_mode='rgb', shuffle=False, class_mode=None,
                                                  batch_size=1)  # , class_mode='categorical') # wird im moment noch nicht benutzt  # ToDo: use color_mode='grayscale'
    return test_generator


# Train the model
def train_model(model, train_generator, valid_generator):
    # used to save checkpoints during training after each epoch
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # simple early stopping
    es_val_loss = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)
    es_val_acc  = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.1, mode='max', restore_best_weights=True)  # val_acc has to improve by at least 0.1 for it to count as an improvement
    callbacks_list = [checkpoint, es_val_acc]

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    train_class_weights = dict(zip(np.unique(train_generator.classes), class_weights))

    print("train_class_weights: \t" + str(train_class_weights))

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, class_weight=train_class_weights,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list,

    return r, model


# Save the models and weight for future purposes
def save_model(model, detailed_model_name):
    model_directory = "dataset/saved_model/"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_directory + detailed_model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_directory + detailed_model_name + ".h5")
    print("\nSaved model to disk")


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()

    # load model that uses transfer learning
    model_, model_name = models.create_pretrained_model_densenet121()

    # load model that uses custom architecture
    # model_, model_name = models.create_custom_model_1d_cnn()
    # model_, model_name = models.create_custom_model_2d_cnn_v2()

    if "1d" in model_name:
        train_path = '../training/'
        ecg_leads, ecg_labels, fs, ecg_names = load_references(train_path)  # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name  # Sampling-Frequenz 300 Hz
        X_train, X_test, y_train, y_test = prep.train_test_split_ecg_leads(ecg_leads, ecg_labels)
        history = model_.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
        model = model_
    else:
        # load image data
        train_generator, valid_generator = load_training_images()
        # Train the model
        history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    # load right model for classification problem
    if binary_classification is True:
        model_name = model_name + '_two_classes'
    else:
        model_name = model_name + '_four_classes'

    detailed_model_name = timestr \
                          + "-" + model_name \
                          + "-num_epochs_" + str(epochs) \
                          + "-batch_size_" + str(batch_size) \
                          + "-image_size_" + str(image_size) \
                          + "-acc_" + str(round(history.history['accuracy'][-1], 4)) \
                          + "-val_acc_" + str(round(history.history['val_accuracy'][-1], 4))

    if save_trained_model:
        save_model(model, detailed_model_name)
        # plots.plot_model_structure(model, detailed_model_name)

    # View the structure of the model
    #  model_.summary()

    plot_directory = "dataset/plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Plot the model accuracy graph
    plots.plot_training_history(history, plot_directory + detailed_model_name)
    # Plot the model accuracy and loss metrics
    plots.plot_metrics(history, plot_directory + detailed_model_name)
