import tensorflow as tf


# import matplotlib.pyplot as plt

# in the case that the model trains for extra epochs with different losses,
# you can stop the training at a desired accuracy (here, 95%) using callbacks.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

# loads the fashion MNIST dataset. Calling load_data on the object gives two sets of two lists
# (graphics - show clothing items and their labels)
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# prints the training_image[0] - gives back an array of pixels
# not sure what printing the training label does? I think it gave back a 9.
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# normalizes the range of the pixels - varied from 0 to 255, this makes it 0 to 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# model is defined the Flatten layer makes the input layer (a square matrix, 28 x 28) into a single vector (784 x 1).
# otherwise, there would be an error. you would need 28 layers of 28 neurons. But flatten does the flattening for
# you. the middle layer neurons can be changed. more can give more accuracy, but slower computation. there is a point
# of diminishing return though.
# the final layer's num of neurons corresponds with the number of classes you're classifying (0 through 9 - 10 neurons)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model by compiling it
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(training_images, training_labels, epochs=5)

# test the model on test data
model.evaluate(test_images, test_labels)

# EXERCISES
# creates a set of classifications for each of the test images
classifications = model.predict(test_images)

# prints out a list of numbers, then a label. The list of numbers represents the probabilities
# of the image being each label. The highest probability is on the 9th label, and print(test_labels[0]) prints 9.
# makes sense!
print(classifications[0])
print(test_labels[0])
