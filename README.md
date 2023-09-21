# Workshop-2-CNN-Model-for-CIAFAR10

## AIM:
To develop a classification model for cifar10 data set using convolution neural network.

## ALGORITHM:
1. Import the required packageas and import dataset using the give line: from tensorflow.keras.datasets import cifar10

2. Split the dataset into train and test data and scale their values for reducing the model complexity.

3. Use the onhot encoder to convert the output of train data and test data into categorical form.

4. Build a model with convolution layer, maxpool layer and flatten it. The build a fully contected layer.

5. Complie and fit the model. Train it and check with the test data.

6. Check the accuracy score and make amendments if required.

### Neural Network Model:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/00d7d420-29c2-48d8-9c51-eec45be33810)

## PROGRAM:

#### 1. Write a python code to load the CIFAR-10 dataset
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

#### 2. Convert the output to one-hot encoded array
```python
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[1281]
plt.imshow(single_image,cmap='gray')
y_train_onehot[1209]
```

#### 3. Create a sequential model with appropriate number of neurons in the output layer, activation function and loss function
```python
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(180,activation="relu"))
model.add(layers.Dense(125,activation="relu"))
model.add(layers.Dense(175,activation="relu"))
model.add(layers.Dense(100,activation="relu"))
model.add(layers.Dense(75,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train ,y_train_onehot, epochs=3,
          batch_size=200,
          validation_data=(X_test,y_test_onehot))
```

#### 4. Plot iteration vs accuracy and iteration vs loss for test and training data
```python
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
```

#### 5. Training the model to get more than 80% accuracy
```python
x_test_predictions = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```


## OUTPUT:
### Model summary:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/d130e905-d2dc-4741-9c1e-a1b5c4621cf2)

### Model accuracy vs val_accuracy:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/40cd8460-3663-4560-bdfa-a00c5b3b3173)

### Model loss vs val_loss:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/b59a287d-925c-4c39-a083-3991faadebb9)

### Confusion matrix:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/a7aa6155-17b0-4d50-b48d-3697e0279b25)

### Classification report:
![image](https://github.com/Aashima02/Workshop-2---CNN-Model-for-CIAFAR10/assets/93427086/a4cda317-091f-41d0-89fc-2828b9648414)


## RESULT:
Thus we have created a classification model for Cifar10 dataset using the above given code.
