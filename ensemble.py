import tensorflow as tf
from tensorflow import keras as K
import efficientnet.keras as efn
from keras.applications import MobileNet, DenseNet201


def input_generator(gen1, gen2):
    x1 = gen1[0]
    x2 = gen2[0]
    y = gen1[1]

    return [x1, x2], y


train_path_filtered = 'filter_gaussian/train/'
valid_path_filtered = 'filter_gaussian/valid/'
test_path_filtered = 'filter_gaussian/test/'
train_path = 'images/train/'
valid_path = 'images/valid/'
test_path = 'images/test/'

##  Data with high pass filter
train_batch_filtered = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = train_path_filtered, target_size = (224,224), batch_size = 10)
valid_batch_filtered = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = valid_path_filtered, target_size = (224,224), batch_size = 10)
test_batch_filtered = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = test_path_filtered, target_size = (224,224), batch_size = 10, shuffle = False)

## Original images
train_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = train_path, target_size = (224,224), batch_size = 10)
valid_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = valid_path, target_size = (224,224), batch_size = 10)
test_batch = ImageDataGenerator(preprocessing_function = K.applications.mobilenet.preprocess_input).flow_from_directory(
    directory = test_path, target_size = (224,224), batch_size = 10, shuffle = False)


## For these two we should be able to use the same ImDataGen we already used.
train_input1 = train_batch
train_input2 = train_batch_filtered
validate_input1 = valid_batch
validate_input2 = valid_batch_filtered

## This will be the actual input and labels for the model
train_combined_generator = map(input_generator, train_input1, train_input2)
validate_combined_generator = map(input_generator, validate_input1, validate_input2)




## Not urgent, but lets make these calls more consistent with each other so it looks cleaner
x = tf.K.applications.mobilenet( weights = 'imagenet' ,
                                include_top = False,
                                 input_shape = (224,224,3))

y = efn( weights = 'imagenet',
         include_top = False,
         input_shape = (224,224,3))


z = K.applications.DenseNet201(     weights = 'imagenet',
                                    include_top = False,
                                    input_shape = (224, 224, 3))


## TODO --->>>> Make sure all layers in those models are frozen

combinedinput = concatenate([x.output, y.output, z.output])

classifier = MAKE FULLY CONNECTED LAYERS AND CLASS. Layers here

model = Model(inputs = [x.input, y.input, z.input], outputs = classifier)

model.compile(hyperparams here)

model.fit(x=combined_generator[0], y=combined_generator[1],
            validation_data = (validate_combined_generator[0], validate_combined_generator[1]),
            epochs=100, batch_size=10)
