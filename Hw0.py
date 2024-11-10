import tensorflow as tf

#test tensorflow neural netowrk with built in dataset
mnist = tf.keras.datasets.mnist

#loading and preparing the data
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0 #pixel value in images range from 0 to 255 so scaling to a range of 0 to 1

#using the Sequential model which stacks layers. Each layer has one input tensor and one output tensor
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(10)
])

#For each example, the model returns a vector of logits or log-odds
predictions = model(x_train[:1]).numpy()

#softmax function converts these logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

#define a loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
loss_fn(y_train[:1],predictions).numpy()

#configure and compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#train model
model.fit(x_train,y_train,epochs =5)

#evaluate model
model.evaluate(x_test, y_test, verbose=2)