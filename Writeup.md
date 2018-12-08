[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##
This project involved the building of a fully convolutional network (FCN) to which was trained to identify a target individual from a simulated drone camera feed.

The model is built within Tensorflow and Keras, and was trained using __*Udacity GPU Workspace*__

[image_0]: ./docs/misc/keras-tensorflow-logo.jpg
![alt text][image_0] 

# Architecture

The FCN is built to be able to segment objects within the video stream. This means that each pixel in the image needs to be labeled. Fully convolutional networks are capable of this via a process called semantic segmentation. 

Semantic segmentation allows FCNs to preserve spatial information throughout the network.

[image_1]: ./docs/misc/ml2.png
![alt text][image_1] 

# Fully Convolutional Networks (FCN)

A fully convolutional net tries to learn representations and make decisions based on local spatial input. Appending a fully connected layer enables the network to learn something using global information where the spatial arrangement of the input falls away and need not apply.

# Encoder

The encoder layers the model to gain a better understanding of the characeristics in the image, building a depth of understanding with respect to specific features and thus the 'semantics' of the segmentation. The first layer might discern colours and brightness, the next might discern aspects of the shape of the object, so for a human body, arms and legs and heads might begin to become successfully segmented. Each successive layers builds greater depth of semantics necessary for the segmentation.

# 1x1 Convolution Layer

The 1x1 convolution layer is a regular convolution, with a kernel and stride of 1. Using a 1x1 convolution layer allows the network to be able to retain spatial information from the encoder. The 1x1 convolution layers allows the data to be both flattened for classification while retaining spatial information.

# Decoder

The decoder section of the model can either be composed of transposed convolution layers or bilinear upsampling layers.

The transposed convolution layers the inverse of regular convolution layers, multiplying each pixel of the input with the kernel.

The decoder block mimics the use of skip connections by having the larger decoder block input layer act as the skip connection. It calculates the separable convolution layer of the concatenated bilinear upsample of the smaller input layer with the larger input layer.

Each decoder layer is able to reconstruct a little bit more spatial resolution from the layer before it. The final decoder layer will output a layer the same size as the original model input image, which will be used for guiding the quad drone.

# Skip Connections

Skip connections allow the network to retain information from prior layers that were lost in subsequent convolution layers. 

# Model 

[image_2]: ./docs/misc/2.png

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc1 = encoder_block(inputs, 32, 2)
    enc2 = encoder_block(enc1, 64, 2)
    enc3 = encoder_block(enc2, 128, 2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    one_x_one = conv2d_batchnorm(enc3,32,kernel_size = 1, strides = 1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec1 = decoder_block(one_x_one,enc2,128)
    dec2 = decoder_block(dec1,enc1,64)
    x = decoder_block(dec2,inputs,32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

# Hyperparameters
```python
learning_rate = 0.0008
batch_size = 100
num_epochs = 40
steps_per_epoch = 200
validation_steps = 50
workers = 2
```
# Training

The model was trained using Udacity GPU Workspace.

# Performance

final score = 0.406304667197

final IoU = 0.551718969142

