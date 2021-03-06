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
[test_wrong_rgb]:  ./examples/test_wrong_rgb.png "test_wrong_rgb"
[test_wrong_YUV]: ./examples/test_wrong_YUV.png "test_wrong_YUV"

[test_new_images]: ./examples/test_new_images.png "test_new_images"
[test_new_images_prob]: ./examples/test_new_images_prob.png "test_new_images_prob"
[intermediate_output_visualisation]: ./examples/intermediate_output_visualisation.png "intermediate_output_visualisation"

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

![alt text][image_examples_top] 

![alt text][image_examples_bottom] 

The complete visualisation with 10 samples per class is shown as below:

Train data: 
 ![alt text][train_examples] 

Test data: 
![alt text][test_examples] 

The label statistics in terms of percentage is shown as below and it can be seen that the different class distribution is quite equal btween training and testing data.
  ![label_stats][label_stats] 


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
| LeNet         | (pixel - 128)/ 128 |0.997                      |0.780                  |-               |
| LeNet         | YUV                |0.997                      |0.915                  |-               | 
| LeNet         | Y channel only     |0.991                      |0.941                  |-               |
|---------------|--------------------|---------------------------|-----------------------|----------------|
| DenseNet      | RGB                |1.000                      |0.988                  |0.975           |
| DenseNet      | RGB (poly lr, 500 epochs) |1.000               |0.990                  |0.982           |
| DenseNet      | YUV                |1.000                      |0.998                  |**0.988**       |
| DenseNet      | Y channel only     |1.000                      |0.993                  |**0.989**       |
|---------------|--------------------|---------------------------|-----------------------|----------------|

If an iterative approach was chosen:

(1) The LeNet architecture was tried because it's the easiest with already implemented. The network didn't achieve good validation accuracy due to its simple architecture with limited parameters.

(2) The Densenet was tried later because it has won the best paper award in CVPR2017. The DenseNets have several advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the numer of parameters.

(3) The learning rate is reduced from the originally 0.1(for CIFAR10 dataset) to 0.5.

(4) Poly learning rate 
```python
learning_rate = train_params['initial_learning_rate'] * (1 - epoch / n_epochs) ** 0.9
```

##### 5. Wrongly classified images in test set

The wrongly classfied images in test set in shown as below (*on the right is the predicted image example in the training set*):

*RGB image as input:*

![test_wrong_rgb][test_wrong_rgb]

*YUV image as input:*
![test_wrong_YUV][test_wrong_YUV]

As it can be seen from above that the mistaken cases made by input transformed into YUV space makes more sense: comparing with the correct class visualisation on the right, the mistakenly predicted labels on the left are visually very similar.


### Test a Model on New Images

#### New images and results:

 I have chosen a great variety traffic signs on the web: (1) standardised German traffic signs that I found from the [wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany); (2) standardize [Chinese traffic signs](http://www.bjjtgl.gov.cn/jgj) and (3) some American traffic signs.
 
Here are the results of the prediction:

![test_new_images][test_new_images]

#### Result analysis

Plot the prediction probability using ``tf.nn.top_k`` we can see the visualisation as belows. It can be seen that it achieves 100% prediction rate if it is from the sets of standrdised German traffic signs from wikipedia. But if the traffic signs categories never appeared in the training categories, it will generate noisy softmax prediction

![test_new_images_prob][test_new_images_prob]



### Visualizing the Neural Network 
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below if the visualisation of the first conv layer output. The first conv layer output obvisouly detect edges etc. features.

![intermediate_output_visualisation][intermediate_output_visualisation]

That's it, everything for the project two. 

THE END!