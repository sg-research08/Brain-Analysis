import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, Average
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model, clone_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D
print("TensorFlow Version:", tf.__version__)

# Adjust the paths according to requirement
base_dir = "Alzheimer_s Dataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)


os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))

WORK_DIR = './dataset/'

# Change according to requirement of dataset
CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

IMG_SIZE = 224
IMAGE_SIZE = [224, 224]
DIM = (IMG_SIZE, IMG_SIZE)

work_dr = IDG(rescale = 1./255)

SAMPLE_PERCENTAGE=100       # Modify according to requirement 20% / 40% / 60% / 80% / 100%
OUTPUT_DIR = '/path/to/output/sampled'
original_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=7000, shuffle=False)

# Iterate through each subfolder and sample images
for i, subfolder in enumerate(original_data_gen.class_indices.keys()):
    subfolder_path = os.path.join(WORK_DIR, subfolder)
# Get the list of all images in the subfolder
    all_images = [image for image in os.listdir(subfolder_path) if image.endswith(('.jpg', '.jpeg', '.png'))]

    # Calculate the number of images to sample based on the percentage
    num_images_to_sample = int(len(all_images) * (SAMPLE_PERCENTAGE / 100))

    # Randomly select 20% of the images
    sampled_images = random.sample(all_images, num_images_to_sample)

    # Create the output subfolder if it doesn't exist
    output_subfolder = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Copy the sampled images to the output subfolder
    for image in sampled_images:
        image_path = os.path.join(subfolder_path, image)
        output_path = os.path.join(output_subfolder, image)
        shutil.copy(image_path, output_path)

# Create a generator for the sampled data
sampled_data_gen = work_dr.flow_from_directory(directory=OUTPUT_DIR, target_size=DIM, shuffle=False)

#Retrieving the data from the ImageDataGenerator iterator

train_data, train_labels = sampled_data_gen.next()

#Getting to know the dimensions of our dataset

print(train_data.shape, train_labels.shape)

#Performing over-sampling of the data, since the classes are imbalanced

sm = SMOTE(random_state=42)

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print(train_data.shape, train_labels.shape)

#Splitting the data into train, test, and validation sets

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

"""### Constructing a Convolutional Neural Network Architecture"""

def construct_model1(act='elu'):

    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(16, 3, activation=act, padding='same'),
        MaxPool2D(),
        Conv2D(16, 3, activation='elu', padding='same'),
        Conv2D(16, 3, activation='elu', padding='same'),
        Conv2D(16, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(32, 3, activation='elu', padding='same'),
        Conv2D(32, 3, activation='elu', padding='same'),
        Conv2D(32, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(64, 3, activation='elu', padding='same'),
        Conv2D(64, 3, activation='elu', padding='same'),
        Conv2D(64, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(128, 3, activation='elu', padding='same'),
        Conv2D(128, 3, activation='elu', padding='same'),
        Conv2D(128, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='elu'),
        BatchNormalization(),
        Dropout(0.7),
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='elu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ], name = "cnn_model1")

    return model

def construct_model2(act='elu'):

    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(16, 3, activation=act, padding='same'),
        MaxPool2D(),
        Conv2D(16, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(32, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(64, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(128, 3, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='elu'),
        BatchNormalization(),
        Dropout(0.7),
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='elu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ], name = "cnn_model2")

    return model

#Defining a custom callback function to stop training our model when accuracy goes above 99%

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True

my_callback = MyCallback()

#EarlyStopping callback to make sure model is always learning
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

#Defining other parameters for our CNN model

model1 = construct_model1()
model2 = construct_model2()

models = [model1, model2]
model_input = Input(shape=(*IMAGE_SIZE, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name='ensemble')

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'),
           tfa.metrics.F1Score(num_classes=4),
          tf.metrics.Precision(),
          tf.metrics.Recall()]

CALLBACKS = [my_callback]

ensemble_model.compile(optimizer='adam',        # Modify optimizers RMSprop/ Adagrad/ SGD
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)

ensemble_model.summary()

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_accuracy', this_max=True):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                print("Save best weight")

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                print("Save best weight")
best_model= SaveBestModel()
#Fit the training data to the model and validate it using the validation data
EPOCHS = 100

history = ensemble_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), callbacks=[CALLBACKS, best_model], epochs=EPOCHS)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy', weight='bold', fontsize=12)
plt.ylabel('Accuracy', weight='bold', fontsize=12)
plt.xlabel('Epoch', weight='bold', fontsize=12)
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("Model Accuracy", dpi=500)
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', weight='bold', fontsize=12)
plt.ylabel('Loss', weight='bold', fontsize=12)
plt.xlabel('Epoch', weight='bold', fontsize=12)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("Model Loss", dpi=500)
plt.show()
#auc
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC', weight='bold', fontsize=12)
plt.ylabel('Loss', weight='bold', fontsize=12)
plt.xlabel('Epoch', weight='bold', fontsize=12)
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("Model AUC", dpi=500)
plt.show()

#Evaluating the model on the data

#train_scores = model.evaluate(train_data, train_labels)
#val_scores = model.evaluate(val_data, val_labels)
test_scores = ensemble_model.evaluate(test_data, test_labels)

#print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
#print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))

#Predicting the test data

pred_labels = ensemble_model.predict(test_data)

#Print the classification report of the tested data

#Since the labels are softmax arrays, we need to roundoff to have it in the form of 0s and 1s,
#similar to the test_labels
def roundoff(arr):
    """To round off according to the argmax of each predicted label array. """
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))

#Plot the confusion matrix to understand the classification in detail

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

plt.figure(figsize=(15, 9), dpi=500, facecolor='w', edgecolor='k')

ax = sns.heatmap(conf_arr, cmap='copper', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)

plt.title('Confusion Matrix', weight='bold', fontsize=12)
plt.xlabel('Predicted Class', weight='bold', fontsize=12)
plt.ylabel('True Class', weight='bold', fontsize=12)
plt.savefig("Confusion matrix 4 class", dpi=500)
plt.show(ax)