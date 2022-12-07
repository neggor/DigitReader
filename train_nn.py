''' For google colab...
!pip install -q tflite-model-maker
!pip install -q pycocotools
!pip install -q tflite-support
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.interpolation import rotate

def convert_to_RGB(data):
    return tf.image.grayscale_to_rgb(tf.constant(np.expand_dims(data, -1))).numpy()

def rotate_and_reduce(img):
    size = np.random.randint(12, 26, 1)[0]
    while True:
        theta = np.random.randint(1, 365, 1)[0]
        if theta < 90 or theta > 270:
            break
    img = rotate(img, theta)
    reduced_im = cv2.resize(img, (size, size))
    placeholder = np.zeros((28, 28, 3), dtype= np.uint8)

    ps1 = np.random.randint(0, int((28-size)/2), 1)[0]
    ps2 = np.random.randint(0, int((28-size)/2), 1)[0]
    placeholder[ps1:(ps1 + size), ps2:(ps2 + size), :] = reduced_im
    return placeholder

def invert_color(images, labels, index):
    return (np.invert(images[index]), labels[index])

def max_saturation(images, labels,  index):
    return ((images[index].astype(bool).astype(float) * 255).astype(np.uint8), labels[index])

def add_noise(images, labels, index):
    noise_level = np.random.randint(2, 6, index.shape[0]) / 10
    return (np.random.poisson(images[index] + np.random.random(images[index].shape) \
        * np.expand_dims(noise_level, (1, 2, 3)) * 255).astype(np.uint8), labels[index])

def amplify(n_loops, data, labels):
    for n in range(n_loops):
        new_data = []
        new_labels = list(labels)
        # first rotate and reduce half of the images 2 times
        for i in range(2):
            index = np.random.randint(0, data.shape[0], int(data.shape[0] / 2))
            for j in index:   
                new_data.append(rotate_and_reduce(data[j]))             
                new_labels.append(labels[j])

        new_data = np.concatenate((data, np.array(new_data)), 0)
        new_labels = np.array(new_labels)
        

        index = np.random.randint(0, data.shape[0], int(data.shape[0] / 4))
        invert_data, invert_labels = invert_color(new_data, new_labels, index)

        index = np.random.randint(0, data.shape[0], int(data.shape[0] / 10))
        maxsat_data, maxsat_labels = max_saturation(new_data, new_labels, index)
        
        index = np.random.randint(0, data.shape[0], int(data.shape[0] / 4))
        noisy_data, noisy_labels = add_noise(new_data, new_labels, index)

        data = np.concatenate((new_data, invert_data, maxsat_data, noisy_data), axis = 0)
        labels = np.concatenate((new_labels, invert_labels, maxsat_labels, noisy_labels), axis = 0)

        print(data.shape)
        print(labels.shape)
    return data, labels


# Load data:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Make RGB:
x_train = convert_to_RGB(x_train)
x_test = convert_to_RGB(x_test)

# Data augmentation:
_x_train, _y_train = amplify(data = x_train, labels = y_train, n_loops = 2)
_x_test, _y_test = amplify(data = x_test, labels = y_test, n_loops = 2)

num_classes = 10
input_shape = (28, 28, 1)

model = tf.keras.Sequential(
    [   
        tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),
        tf.keras.layers.Rescaling(
            1./255),
        tf.keras.Input(shape=input_shape),
        #tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    _x_train, _y_train,
    epochs=10,
    validation_data=(_x_test, _y_test),
    batch_size = 128 
)


model.save('Point_estimate_digits_model')

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('Point_estimate_digits_model')
tflite_model = converter.convert()

# Save the model.
with open('Model_RGB_point_estimate.tflite', 'wb') as f:
  f.write(tflite_model)

plt.plot(history['val_loss'])
plt.show()