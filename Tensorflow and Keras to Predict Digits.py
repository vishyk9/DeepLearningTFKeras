
# coding: utf-8


import tensorflow as tf
mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #flatten the image to get a array
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) ##default function
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #output layer

model.compile(optimizer ='adam',
             loss ='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs =3)



val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)

predictions = model.predict([x_test])
