[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##
This project involved the building of a fully convolutional network (FCN) to which was trained to identify a target individual from a simulated drone camera feed.

The model is built within Tensorflow and Keras, and was trained using __*Udacity GPU Workspace*__

[image_0]: ./docs/misc/keras-tensorflow-logo.jpg
![alt text][image_0] 

# Architecture
## Fully Convolutional Network over Fully Connected Network
Fully Connected Network works good for classification problems such as "is that a burger in image?" while when answers such as "where is the burger in the image?" Fully Connected Network doesn't work well because flattening layer at the output layer. This leads to spatial information loss in data. These issues can be dealt with Fully Convolutional Layer.
The FCN is built to be able to segment objects within the video stream. This means that each pixel in the image needs to be labeled. Fully convolutional networks are capable of this via a process called semantic segmentation.
Semantic segmentation allows FCNs to preserve spatial information throughout the network.

[image_1]: ./docs/misc/ml2.png
![alt text][image_1] 

# Fully Convolutional Networks (FCN)

A Fully Convolutional Networks have a convolution layer, 1x1 convolution layer and a decoder section made of reversed convolution layers. Instead of a final fully connected layer, like a Fully Connected Layer, every layer in an Fully Convolutional Network is a fully convolutional layer. A fully convolutional net tries to learn representations and make decisions based on local spatial input

## Encoder
The encoder block extracts the feature from an image. Encoding in general narrows down the scope by looking closely at some picture and loose the bigger picture as a result. Encoder block contains convolutional layer and also might contain max pooling layer. The extracted features from image by are later used by decoder.
Separable convolution layers are a convolution technique for increasing model performance by reducing the number of parameters in each convolution. A spatial convolution is performed, followed with a depthwise convolution. Separable convolutions stride the input with only the kernel, then stride each of those feature maps with a 1x1 convolution for each output layer, and then add the two together. This technique allows for the efficient use of parameters.

The encoder layers the model to gain a better understanding of the characeristics in the image, building a depth of understanding with respect to specific features and thus the 'semantics' of the segmentation. The first layer might discern colours and brightness, the next might discern aspects of the shape of the object, so for a human body, arms and legs and heads might begin to become successfully segmented. Each successive layers builds greater depth of semantics necessary for the segmentation. However, the deeper the network, the more computationally intensive it becomes to train.

## 1x1 Convolution Layer
As discussed earlier as well, flattening looses the spatial information in fully coonected layer because no information about the location of the pixel is preserved.
The 1x1 convolution layer is a regular convolution, with a kernel and stride of 1. Using a 1x1 convolution layer allows the network to be able to retain spatial information from the encoder. The 1x1 convolution layers allows the data to be both flattened for classification while retaining spatial information.
## Decoder

The decoder section of the model can either be composed of transposed convolution layers or bilinear upsampling layers.
The transposed convolution layers the inverse of regular convolution layers, multiplying each pixel of the input with the kernel.

Bilinear upsampling is similar to 'Max Pooling' and uses the weighted average of the four nearest known pixels from the given pixel, estimating the new pixel intensity value. Although bilinear upsampling loses some details it is much more computationally efficient than transposed convolutional layers.

The decoder block mimics the use of skip connections by having the larger decoder block input layer act as the skip connection. It calculates the separable convolution layer of the concatenated bilinear upsample of the smaller input layer with the larger input layer.

Each decoder layer is able to reconstruct a little bit more spatial resolution from the layer before it. The final decoder layer will output a layer the same size as the original model input image, which will be used for guiding the quad drone.

# Skip Connections

Skip connections allow the network to retain information from multiple resolution scale, as a result more precise segmentation decisions. The encoder narrows down the scope by looking closely at some feature and loose the bigger picture as a result. So, even if we decode the output of encoder back to image some information has been lost. Skip connection is the way to retaing those information easily by connecting the output of one layer to non-adjacent using element-wise addition operation.
## Disadvantages of adding too many skip connection
Explosion in the size of our model.

# Model Implementation and Explanation

[image_2]: ./docs/misc/2.png
![alt text][image_2]

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
The FCN model used for the project contains three encoder block layers, a 1x1 convolution layer, and three decoder block layers.

The first convolution uses a filter size of 32 and a stride of 2, while the second convolution uses a filter size of 64 and a stride of 2 and the third uses a filter size of 128 and a stride of 2. These convolutions used same padding. The padding and the stride of 2 cause each layer to halve the image size, while increasing the depth to match the filter size used, finally encoding the input within the 1x1 convolution layer uses a filter size of 32, with the standard kernel and stride size of 1.

The first decoder block layer uses the output from the 1x1 convolution as the small input layer, and the first convolution layer as the large input layer, thus mimicking a skip connection. A filter size of 128 is used for this layer.

The second decoder block layer uses the output from the first decoder block as the small input layer, and the original image as the large input layer, again mimicking the skip connection to retain information better through the network. This layer uses a filter size of 64 and the third block layer x uses filter size of 32.

The output convolution layer x applies a softmax activation function to the output of the second decoder block.

# Hyperparameters
The optimal hyperparamers i found are:
```python
learning_rate = 0.0008
batch_size = 100
num_epochs = 40
steps_per_epoch = 200
validation_steps = 50
workers = 2
```
Hyperparameters were tuned manually by inspection method i.e, trying out different value, checking out the performance and adjusting again.  
The range of learning rate was: 0.01-0.0009.

Regarding the batch_size it was calculated based on initial dataset size of 3000 images by estimating around 200 steps_per_epoch. Therefore, the batch_size was kept equal to 100. Another reason behind this value is save computation time to train the nework. Eventually, this number could be increased in order to avoid floatuation of error through epochs.

The chosen number of epochs was 40. The adopted procedure was recording 40 epochs each time and save the weights according to error keep decreasing and the network could converge to a local minimum.

# Training

The model was trained using Udacity GPU Workspace.

# Performance

final score = 0.406304667197

final IoU = 0.551718969142

# Future Enhancements
This model was trained on people, however, it could be used to train on any other objects of interest such as dog, cat, Car, Horse etc. The model could carefully be added with more convolution layers on both encoder and decoder side could conceivably be trained on any set of labelled data large enough. The learning rate can be coded to decrease over time according to the differential of the validation loss over time and to undertake a performance-based search.



