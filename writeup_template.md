# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_examples_top]: ./examples/top_examples.png "Top Examples"
[image_examples_bottom]: ./examples/bottom_examples.png "Bottom Examples"
[train_examples]: ./examples/train_examples.png "Train Examples"
[test_examples]: ./examples/test_examples.png "Test Examples"
[label_stats]: ./examples/label_stats.png "label_stats"
[image_YUV]: ./examples/YUV_image.png "YUV_image"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"


[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stevenwudi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
I first plot the traffic sign classes with the most number of training examples and the classes with the least number of training examples.

<!-- ![alt text][image_examples_top] -->

<!-- ![alt text][image_examples_bottom] -->

The complete visualisation with 10 samples per class is shown as below:

Train data: 
<!--  ![alt text][train_examples] -->

Test data: 
<!--  ![alt text][test_examples] -->

The label statistics in terms of percentage is shown as below and it can be seen that the different class distribution is quite equal btween training and testing data.
<!--  ![label_stats][label_stats] -->


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


#### First attempt

1) preprcoessing image data: (pixel - 128)/ 128

2) Model architecture: LeNet.

The learning rate set from 0.001 and number of epochs sets as 100.
With this preprocessing step, I only achieved very low validation accuracy of ~0.78 and training accuracy ~0.997. This is a clear sign of overfitting.

#### Second attempt

Better image preprocessing: change the RGB image to YUV color space, we have a validation accuracy of **~0.915** and training accuracy **~0.997** which is still under the required 0.93 validation accuracy.

If I use only Y channel space, we have a validation accuracy of **~0.941** and training accuracy **~0.991**. Now it's the first time the model has surpassed the required accuracy, hourray!

With data normalisation scheme:  X=(X-X_mean)/(X_std), I further boost the validation accuracy to **~0.950** and test accuracy of **~0.933**.  Data normalisation help the network to learn better.

The following are the image examples of YUV:

![image_YUV][image_YUV]

 #### Third attempt
 
 Better model architecture: (1) DenseNet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)


 ##### 2.  final model densenet

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs size 32x32	|
| Dense Block			| growth rate 12								|
| Transition Block     	| avg pool,  outputs size 16x16 	       		|
| Dense Block	        | growth rate 12	   							|
| Transition Block 		| avg pool,  outputs size 8x8 		  			|
| Dense Block			| growth rate 12								|
| Transition Block 		|avg pool,  outputs 4x4 		     			|
| FC		            | Softmax activation, output 43					|

With data normalisation scheme:  X=(X-X_mean)/(X_std), I further boost the validation accuracy to **~0.988** and test accuracy of **~0.979**.  With no data augmentation.

#### 3. Training details. 

optimizer: tensorflow MomentumOptimizer,

batchsize: 128,

number of epochs: 100,

learning rate: 0.1, divide by 10 on 50th and 75th epoch. (Interestingly, intial learning rate is too large for the network: high accuracy (>0.95) for training set and low (~0.1) for the validation set. The validation accuracy only start to increase after 50th epoch. I guess it's an indication of the large learning rate chosen).

#### 4. Steps in a table

My final model results were:

| Model         | input              |training accuracy          | validation  accuracy  | test  accuracy |
| ------------- |:------------------:|:-------------------------:|:---------------------:|:--------------:|
|---------------|--------------------|---------------------------|-----------------------|----------------|
| LeNet         | (pixel - 128)/ 128 |0.997                      |0.78                   |-               |
| LeNet         | YUV                |0.997                      |0.915                  |-               | 
| LeNet         | Y channel only     |0.991                      |0.941                  |-               |
|---------------|--------------------|---------------------------|-----------------------|----------------|
| DenseNet      | YUV                |1.000                      |0.988                  |0.979           |
| DenseNet      | Y channel only     |1.000                      |0.988                  |0.971           |
| DenseNet      | YUV (augmentated)  |1.000                      |0.988                  |0.971           |
|---------------|--------------------|---------------------------|-----------------------|----------------|

If an iterative approach was chosen:
(1) The LeNet architecture was tried because it's the easiest with already implemented. The network didn't achieve good validation accuracy due to its simple architecture with limited parameters.
(2) The Densenet was tried later because it has won the best paper award in CVPR2017. The DenseNets have several advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the numer of parameters.
(3) The learning rate is reduced from the originally 0.1(for CIFAR10 dataset) to 0.5.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


