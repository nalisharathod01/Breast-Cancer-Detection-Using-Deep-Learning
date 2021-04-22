#libs here


input1 = orig_images
input2 = trans_images
^^ These will need to be modified to match the map generator I shared in the chat

x = mobilenet

y = efficientnet

z = densenet


^^^ Make sure classification layers are rmoved. Should be as simple as passing "include_top=False" to the model call
Also, make sure all layers are frozen.

combinedinput = concatenate([x.output, y.output, z.output])

classifier = MAKE FULLY CONNECTED LAYERS AND CLASS. Layers here

model = Model(inputs = [x.output, y.output, z.output], outputs = classifier)

model.compile(hyperparams here)
