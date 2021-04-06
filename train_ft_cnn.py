import tensorflow as tf
from tensorflow import keras as K

o_model = K.applications.MobileNetV2()

trainset = K.preprocessing.image_dataset_from_directory("ft_images", class_names=('benign', 'malignant'), image_size=(224,224))

model = K.Sequential()
model.add(o_model)

model.layers.pop()
model.add(K.layers.Dense(2))

class_names = trainset.class_names

learning_rate = 0.001
model.compile(optimizer=K.optimizers.Adam(lr=learning_rate),
                loss = K.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

model.summary()

eval = model.fit(trainset, batch_size=40, epochs=100)

model.save("try")
