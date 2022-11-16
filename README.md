# Object-Detection-using-Deep-Learning-Algorithm-CNN
I have Used Convolutional Neural Network in my model. Convolutional Neural Network (CNN) is a Deep Learning based  algorithm that can take images as input, assign classes for the objects in the image.
Author : Komal Kedari

IDE: Jupyter Notebook

Language: Python 3

Libraries used : Pandas , Numpy , Sk-learn , Tensorflow , Keras

Problem Statement:

Predict the class of an object in an image

Dataset: The dataset used in our model contains 300 images belonging to 4 different classes, which are  Apple, Banana, Orange and mixed 
The dataset downloaded from Kaggle Datasets, which is a collection of datasets ready to use. The dataset 
has divided into Training Set and Testing Set. To train our model, we use images in Training Set, and the Testing set contains 
the images used for performance evaluation. 

Abstract: Object Detection is an emerging technology in the field of Computer Vision and Image Processing that deals with 
detecting objects of a particular class in digital images. It has considered being one of the complicated and challenging tasks in 
computer vision. I have implemented a Deep Learning based approach Convolutional Neural Network (CNN) in this paper. The Proposed approach provides accurate 
results in detecting objects in an image.

Introduction:
Humans are very good at interpreting unstructured data, such as images and audio files. One of the best things about the rise of deep 
learning is that computers are much better at interpreting unstructured data compared to a few years ago. It creates opportunities for 
many new exciting applications like Object detection, Speech recognition, image recognition. Object Detection deals with 
identifying individual objects in an image along with its location. 
Object Detection has two parts-Object Classification and Object Localization. Object Classification deals with classifying the object 
into one of the pre-defined classes. Object Localization deals with distinguishing the object along with its location. 
The main idea of Object detection systems is to construct a model for an object class from a set of training examples. The idea is to 
provide a training data set and test the input image by comparing the objects in the images with the training dataset containing 
images of objects. The output displayed with the objects detected after comparing the input image with the training dataset.

Object detection system takes an image as input. In the pre-processing stage, the image is resized and converted 
into a matrix. The system extract features from an image to recognize a particular object. It classifies the object to a finite set of 
classes. Also, it predicts the location of the object in terms of a bounding box.

Architecture:

1) Input Layer: The process of training data starts with an input image provided by the user. the input image 
converted into a matrix, which may be of the form 300 x 300 x 3 where,3 represents RGB,  

2) Convolutional Layer: The objective of the Convolution Layer is to extract features such as edges, color, gradient orientation from 
the input image. The convolutional operation is carried out with the help of an element called kernel/filter.In my program, I 
have implemented the function Conv2D() as an image is nothing but a 2-dimensional array. We can use Convolution 3D if we
need to implement videos, where the third dimension is time. The convolution operation results in a feature map.

3) ReLU Activation Function: ReLU stands for the Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max (0, 
x). It is applied to the resultant feature map to convert the negative values into positive (0)
    
4) Pooling Layer: The different kinds of pooling layers are max pooling, min pooling, mean pooling, average pooling. We have 
used max-pooling because we need the maximum value pixel from the region of interest,The process of convolution-pooling is executed 3 times in our program.

5) Flattening:Flattening is a process of converting the resulting 2-dimensional arrays of the convolutionpooling operations into a single long continuous linear vector. 
To perform the process of flattening, we implement the flatten () 
function. We flatten our matrix into a vector and feed it into a fully connected layer like a neural network.

6) Fully Connected Layer: The fully connected layer takes the flatted vector and uses them to classify the image into a label and 
passes it to the next layer, With the fully connected layers, we combined these features to create a model. 
After features extracted by the convolution layers and decreased by the pooling layers, the reduced image mapped with the 
subgroup of a fully connected layer, such as the probability of each class in classification tasks. The final output of a fully 
connected layer has the same number of output nodes as the number of classes.

7) Soft Max: An activation function applied to the multiple class classification task is a SoftMax function. The main goal of this 
function is to normalize the output values of the last fully connected layer to target class probabilities, where each value ranges 
between 0 and 1.

Analysis:
    
I have implemented 4 datasets or classes. To built the CNN Model, the training data split into Training Set and Validation Set. Validation Set is a part of 
the training data that can be used to build a model. Validation Set is used to avoid overfitting. The training process consists of 50 
epochs with 30 epochs or steps per epoch, i.e. a loop within a loop. 
After the CNN model built, the message "CNN built successfully" is displayed along with the accuracy and loss values. The values 
displayed are as follows loss: 0.1196 - acc: 0.9440 - val_loss: 1.7738 - val_acc: 0.8167 are displayed. Here, loss and acc are 
applied to training data, while val_loss and val_acc applied to validation data.
