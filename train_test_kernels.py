import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = K.Sequential()
model.add(K.applications.DenseNet201(
                                    weights = 'imagenet',
                                    include_top = False,
                                    input_shape = (224, 224, 3)))
model.add(K.layers.GlobalAveragePooling2D())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(2, activation='softmax'))

model.compile(
        loss = 'categorical_crossentropy',
        optimizer = K.optimizers.Adam(lr=1e-4),
        metrics = ['accuracy'])

model.summary()

import datetime

path = "logs_5/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=path, histogram_freq=0,
                                                          write_grads=False, write_images=False, embeddings_freq=0,
                                                          embeddings_layer_names=None, embeddings_metadata=None,
                                                          embeddings_data=None, update_freq=50)

CheckPoint = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', monitor='val_loss', verbose=0,
                                                               save_best_only=False, save_weights_only=False, mode='auto')
callbacks = [CheckPoint, tensorBoard]

train_path = 'filterx5/train/'
valid_path = 'filterx5/valid/'
test_path = 'filterx5/test/'

train_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = train_path, target_size = (224,224), batch_size = 10)
valid_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = valid_path, target_size = (224,224), batch_size = 10)
test_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = test_path, target_size = (224,224), batch_size = 10, shuffle = False)



eval = model.fit(x=train_batch, validation_data=valid_batch, verbose=2, batch_size=20, epochs=20, callbacks=callbacks)

model.save("model_5")
