import keras.models
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.utils import img_to_array, load_img

from EnowNet.models import (
    le_net_5_model,
    alex_net_model,
    zf_net_model,
    vgg_net_model
)

from EnowNet.detection import detection_model


"""
1./ Load feature extraction layers for the 
    different models
"""
leNet_5 = le_net_5_model()
alex_net = alex_net_model()
zf_net = zf_net_model()
vgg_net = vgg_net_model()
detection = detection_model()

models = [
    leNet_5, alex_net, zf_net, vgg_net, detection]


# index the model which you
# want to study.
# Here, [0] is for leNet_5
model = models[4]


"""
2./ Get list of layers from any model
"""
layer_output = [
    layer.output for layer in model.layers[1:]]

"""
3./ Create a visualization model
"""
visualize_model = keras.models.Model(
    inputs=model.inputs,
    outputs=layer_output)

"""
4./ Load image for prediction
    - Convert image to array
    - Reshape image for passing it into prediction
    - Rescale image
"""
path = "C:/EnowNet_Datasets/bell_pepper_BS/train/disease/0bd0f439-013b-40ed-a6d1-4e67e971d437___JR_B.Spot 3272.JPG"
target_size = (128, 128)

image = load_img(path=path, target_size=target_size)
x = img_to_array(image)
# print(x.shape)

x = x.reshape((1, 128, 128, 3))
x = x/255


"""
5./ Get all layers feature maps for image
"""
feature_maps = visualize_model.predict(x)
# print(len(feature_maps))


"""
6./ Show names of layers available in model
"""
layer_names = [layer.name for layer in model.layers]
# print(layer_names)


"""
7./ Plotting the graph
"""
for layer_names, feature_maps in zip(
        layer_names, feature_maps):
    print(feature_maps.shape)

    if len(feature_maps.shape) == 4:
        channels = feature_maps.shape[-1]
        size = feature_maps.shape[1]

        display_grid = np.zeros(
            (size, size * channels))

        for i in range(channels):
            x = feature_maps[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')

            # Each filter shall be tied into this
            # big horizontal grid below

            display_grid[
                :, i * size: (i + 1) * size] = x

        scale = 20. / channels

        plt.figure(figsize=(scale * channels, scale))
        plt.title(layer_names)
        plt.grid(False)
        plt.imshow(
            display_grid,
            aspect='auto',
            cmap='Accent')

        plt.axis('off')

        plt.show()
