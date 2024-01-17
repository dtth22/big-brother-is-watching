import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from PIL import Image

# Each subdirectory in 'training-data' corresponds to one class.
# 'testing-data' should have the same numbers folders with the same name.

class_names = []
dataset_dir = 'training-data'
for i, type_name in enumerate(sorted(os.listdir(dataset_dir))):
    class_names.append(type_name)
    
# Training and testing images will be downsampled.

normalised_img_width = 32
normalised_img_height = 16
normalised_img_colours = 16

# Look for 'model.keras'.
# If the model doesn't exist, a new model will be created and trained with 'training-data', then saved as 'model.keras'.
if not os.path.isfile('model.keras'):
    train_images = []
    train_labels = []

    # Iterate through all subdirectories in 'dataset_dir', each of which corresponds to a class
    dataset_dir = 'training-data'
    for i, type_name in enumerate(sorted(os.listdir(dataset_dir))):
        type_dir = dataset_dir + '/' + type_name
        for img_name in os.listdir(type_dir):
            # Let's work exclusively with png images for now.
            if not img_name.endswith('png'):
                continue

            # Downsample the image.
            img_dir = type_dir + '/' + img_name
            img = Image.open(img_dir)
            img = img.resize((normalised_img_width, normalised_img_height), Image.LANCZOS)
            img = img.convert('P', palette=Image.ADAPTIVE, colors=normalised_img_colours)

            # The image is appended to train_images as a NumPy array.
            train_images.append(np.asarray(img))
            train_labels.append(i)
        print('Retrieved training data from folder', type_name)

    # train_images and train_labels should be NumPy arrays.
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Images in train_images are stored as NumPy arrays.
    # Each array element is an integer in range [0, normalised_img_colours).
    # We want to work with real numbers in range [0, 1], so we divide the array by (normalised_img_colours - 1).
    train_images = train_images / float(normalised_img_colours - 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(normalised_img_height, normalised_img_width)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=256)

    model.save('model.keras')

model = keras.models.load_model('model.keras')

test_images = []
test_labels = []

# Iterate through all subdirectories in 'dataset_dir', each of which corresponds to a class
dataset_dir = 'testing-data'
for i, type_name in enumerate(sorted(os.listdir(dataset_dir))):
    type_dir = dataset_dir + '/' + type_name
    for img_name in os.listdir(type_dir):
        # Let's work exclusively with png images for now.
        if not img_name.endswith('png'):
            continue

        # Downsample the image.
        img_dir = type_dir + '/' + img_name
        img = Image.open(img_dir)
        img = img.resize((normalised_img_width, normalised_img_height), Image.LANCZOS)
        img = img.convert('P', palette=Image.ADAPTIVE, colors=normalised_img_colours)

        # The image is appended to test_images as a NumPy array.
        test_images.append(np.asarray(img))
        test_labels.append(i)
    print('Retrieved testing data from folder', type_name)

# test_images and test_labels should be NumPy arrays.
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Images in test_images are stored as NumPy arrays.
# Each array element is an integer in range [0, normalised_img_colours).
# We want to work with real numbers in range [0, 1], so we divide the array by (normalised_img_colours - 1).
test_images = test_images / float(normalised_img_colours - 1)

# We evaluate testing data with the model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Each element in predictions is a NumPy array of size len(class_names).
# predictions[i][j] is the predicted probability of the i-th image belonging to the j-th class.
# We will illustrate the first num_images prediction with MatPlotLib.

num_rows = 4
num_cols = 5
num_images = num_rows * num_cols

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
