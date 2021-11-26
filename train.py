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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import plots
import models


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

chkp_filepath = 'model_images'   # Enter the filename you want your model to be saved as
train_path = '../training_complete_6000/images/'                        # Enter the directory of the training images
#valid_path = '../test_images_20'                        # Enter the directory of the validation images

epochs = 30
batch_size = 32
image_size = 512
IMAGE_SIZE = [image_size, image_size]               # re-size all the images to this
save_trained_model = True


def get_num_of_classes():
    return len(glob(train_path + '/*'))


# load image data and convert it to the right dimensions to train the model
def load_training_images():
    train_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # rescale=1./255 to scale colors to values between [0,1]
    train_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size, subset='training') #, class_mode='categorical')
    valid_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size, subset='validation') #, class_mode='categorical')

    return train_generator, valid_generator


# load image data and convert it to the right dimensions to test the model on unseen data
def load_test_images(valid_path):
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, shuffle=True,
                                                  batch_size=batch_size)  # , class_mode='categorical') # wird im moment noch nicht benutzt
    return test_generator

# Train the model
def train_model(model, train_generator, valid_generator):
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]       # used to save checkpoints during training after each epoch     # currently unused

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list,
                # steps_per_epoch=len(trainings_samples),validation_steps=len(validation_samples))

    return r, model


# Save the models and weight for future purposes
def save_model(model, detailed_model_name):
    timestr = time.strftime("%Y%m%d-%H%M%S")

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

    # load image data
    train_generator, valid_generator = load_training_images()

    # load model that uses transfer learning
    model_, model_name = models.create_pretrained_model_densenet121()

    # load model that uses custom architecture
    # model_, model_name = models.create_custom_model_2d_cnn_v2

    # View the structure of the model
    # model_.summary()

    # Train the model
    history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    detailed_model_name = model_name \
                          + "-num_epochs_" + str(epochs) \
                          + "-batch_size_" + str(batch_size) \
                          + "-image_size_" + str(image_size) \
                          + "_" + timestr

    if save_trained_model:
        save_model(model, detailed_model_name)
        plots.plot_model_structure(model, detailed_model_name)

    # Plot the model accuracy graph
    plots.plot_training_history(history)
    # Plot the model accuracy and loss metrics
    plots.plot_metrics(history)
