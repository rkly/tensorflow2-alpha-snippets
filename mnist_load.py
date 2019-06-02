import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


# Recreate the model
def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
  
model = MNIST()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
mnist_test = dataset['test'].map(convert_types).batch(32)

for image, _ in mnist_test:
	predictions = model(image)
	break

# Load the state of the old model
model.load_weights('keras_saved')

for image, _ in mnist_test:
	predictions = model(image)
	print(predictions)
