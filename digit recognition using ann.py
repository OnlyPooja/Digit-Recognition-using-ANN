# program is an example of building and training a neural network (Artificial Neural Network or ANN) for digit recognition using the MNIST dataset


import tensorflow          # for neural network
from tensorflow import keras        #  keras for high-level neural network APIs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

# load the MNIST dataset:

#  It consists of 28x28 pixel grayscale images of handwritten digits (0-9) for training and testing.

(X_train,Y_train),(X_test,Y_test) =keras.datasets.mnist.load_data()
print(X_train)       # X_train and X_test are NumPy arrays containing the pixel values of images.
print(X_test.shape)     # TELLS HOW MANY ROWS
print(Y_train)          # Each element in Y_train represents the true digit label (0-9) for the corresponding image in X_train.

# Tells which image is in particular column
import matplotlib.pyplot as plt
plt.imshow(X_train[0])                 #   is used to display an image, and X_train[0] is the first image in the training dataset.
plt.show()    # this will show the image at particular coloumn

# converting all values from 0 to 255 of images to 0 to 1

X_train = X_train/255          #  Dividing each pixel value by 255 scales the pixel values from the original range of 0-255 to a new range of 0-1
X_test = X_test/255            #  When pixel values are in the 0-1 range, it can be easier for the neural network to learn and converge because it avoids issues related to large input values.
print(X_test[0])

#ANN

model=Sequential()         # creating an empty neural network model with no layers.
# Sequential Model: A sequential model is a linear stack of layers that are executed in a sequential order, one after the other. It is a straightforward way to build a neural network where data flows sequentially from one layer to the next.
# we need 784 networks but we have data in form of 28*28 so we will flatten data first


# Flatten Layer: The layer being added is the "Flatten" layer. It's used to convert multi-dimensional input data into a one-dimensional vector.
# model.add(...): This line adds a new layer to the neural network model.
# one-dimensional array of 28*28 = 784 values.
model.add(Flatten(input_shape=(28,28)))     # will convert the data into 1 d form from its 2-d form

# Dense Layer: The Dense layer is a standard fully connected layer in a neural network. Each neuron in a dense layer is connected to every neuron in the previous layer, making it fully connected.

model.add(Dense(128,activation='relu'))     # first hidden layer of 128

model.add(Dense(32,activation='relu'))      # second layer hidden

#   the output layer with 10 neurons and softmax activation is designed for multi-class classification. The neural network will compute class probabilities for each of the 10 possible digits (0-9), and the class with the highest probability will be considered the predicted digit when making predictions.

model.add(Dense(10,activation='softmax'))   # output layer of 10 layer (0-9)

print(model.summary())


#  code is related to training a neural network for digit recognition using the MNIST dataset.

# a neural network code is responsible for configuring and compiling the neural network model before training
# The sparse_categorical_crossentropy loss is commonly used for multi-class classification problems where the target labels are integers (e.g., 0, 1, 2) rather than one-hot encoded vectors.
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# training
# After the model is trained, the history object will contain information about how the model's performance changed during training, such as training loss, validation loss, and other metrics.
history = model.fit(X_train,Y_train,epochs=10,validation_split=0.2)

y_prob = model.predict(X_test)          #  to generate predictions (probabilities or scores) for a set of input data (X_test) and stores those predictions in the variable y_prob
y_pred = y_prob.argmax(axis=1)          #   is used to convert the predicted probabilities (y_prob) into predicted class labels by selecting the class with the highest probability for each input example
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)           #  The result of accuracy_score(Y_test, y_pred) is a single accuracy score, typically ranging from 0 to 1, where 1 represents perfect accuracy (all predictions are correct), and lower values indicate less accurate predictions
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)
plt.plot(history.history['loss'])    #  It helps you understand whether the model is converging (reducing loss) or if there are signs of overfitting (increasing loss on the validation set).
# loss-> It quantifies the difference between the model's predictions and the actual target values
plt.show()
plt.plot(history.history['val_loss'])       # The validation loss is a crucial metric during the training of machine learning models, especially deep neural networks. It provides insight into how well the model generalizes to data that it has not seen during training.
plt.show()
plt.plot(history.history['accuracy'])       #  This code is used to plot the accuracy (classification accuracy) during the training of a classification machine learning model.
plt.show()
plt.plot(history.history['val_accuracy'])    #  This code is used to plot the validation accuracy during the training of a classification machine learning model
plt.show()
plt.imshow(X_test[4])        # displays 5th image of dataset
plt.show()
print(model.predict(X_test[4].reshape(1,28,28)).argmax(axis=1))           #  It accesses the fifth element