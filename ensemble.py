import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import concatenate

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

train_path_filtered = 'filter_gaussian/train/'
valid_path_filtered = 'filter_gaussian/valid/'
test_path_filtered = 'filter_gaussian/test/'
train_path = 'images/train/'
valid_path = 'images/valid/'
test_path = 'images/test/'

Datagen = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input)

def generate_input(path1, path2):
    gen1 = Datagen.flow_from_directory(directory=path1, target_size=(224,224), batch_size=250, seed=101)
    gen2 = Datagen.flow_from_directory(directory=path2, target_size=(224,224), batch_size=250, seed=101)

    while True:
        x1 = gen1.next()
        x2 = gen2.next()
        yield [x1[0], x1[0], x2[0]], x1[1]


x = K.applications.MobileNet(      weights = 'imagenet' ,
                    include_top = False,
                    input_shape = (224,224,3))
for layer in x.layers:
    layer.trainable = False

y = K.applications.efficientnet.EfficientNetB0( weights = 'imagenet',
                    include_top = False,
                    input_shape = (224,224,3))
for layer in y.layers:
    layer.trainable = False


z = K.applications.DenseNet201(    weights = 'imagenet',
                    include_top = False,
                    input_shape = (224, 224, 3))
for layer in z.layers:
    layer.trainable = False


combinedInput = concatenate([x.output, y.output, z.output])


classifier = K.layers.GlobalAveragePooling2D()(combinedInput)
classifier = K.layers.Dropout(0.5)(classifier)
classifier = K.layers.BatchNormalization()(classifier)
classifier = K.layers.Dense(2, activation='softmax')(classifier)

model = K.Model(inputs = [x.input, y.input, z.input], outputs = classifier)

model.compile(  optimizer=K.optimizers.Adam(),
                loss='binary_crossentropy',
                metrics=['accuracy'])



import datetime
path = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=path, histogram_freq=0,
                                                          write_grads=False, write_images=False, embeddings_freq=0,
                                                          embeddings_layer_names=None, embeddings_metadata=None,
                                                          embeddings_data=None, update_freq=50)

CheckPoint = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', monitor='val_loss', verbose=0,
                                                               save_best_only=False, save_weights_only=False, mode='auto')
callbacks = [CheckPoint, tensorBoard]

with tf.device('/device:GPU:0'):
    model.fit_generator(  generate_input(train_path, train_path_filtered),
                validation_data = generate_input(valid_path, valid_path_filtered),
                steps_per_epoch=5545, epochs=100, verbose=2, callbacks=callbacks,
                validation_steps=1576)
                
model.save('final_model')
