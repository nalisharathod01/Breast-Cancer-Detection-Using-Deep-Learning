#libs here

def input_generator(gen1, gen2):
    x1 = gen1[0]
    x2 = gen2[0]
    y = gen1[1]

    return [x1, x2], y

## For these two we should be able to use the same ImDataGen we already used.
input1 = orig_images
input2 = trans_images

## This will be the actual input and labels for the model
combined_generator = map(input_generator, input1, input2)



x = mobilenet

y = efficientnet

z = K.applications.DenseNet201(     weights = 'imagenet',
                                    include_top = False,
                                    input_shape = (224, 224, 3)))


^^^ Make sure classification layers are rmoved. Should be as simple as passing "include_top=False" to the model call
Also, make sure all layers are frozen.

combinedinput = concatenate([x.output, y.output, z.output])

classifier = MAKE FULLY CONNECTED LAYERS AND CLASS. Layers here

model = Model(inputs = [x.output, y.output, z.output], outputs = classifier)

model.compile(hyperparams here)
