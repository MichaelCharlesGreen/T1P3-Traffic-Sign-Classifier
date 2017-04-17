#**Traffic Sign Recognition** 

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_training_images.png "Random Training Images"
[image1b]: ./examples/histogram_training.png "Training Dataset Histogram"
[image2]: ./examples/grayscale.png "Grayscaling"
[image2b]: ./examples/normalize.png "Normalizing"
[image3]: ./examples/augmented.png "Augmented"
[image4]: ./examples/01.jpg "Traffic Sign 1"
[image5]: ./examples/02.jpg "Traffic Sign 2"
[image6]: ./examples/03.jpg "Traffic Sign 3"
[image7]: ./examples/04.jpg "Traffic Sign 4"
[image8]: ./examples/05.jpg "Traffic Sign 5"
[image9]: ./examples/softmax.png "Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MichaelCharlesGreen/T1P3-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the <span style="color:green"><b>Basic Summary of the Data Set</b></span> cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here is an exploratory visualization of the dataset.

The <span style="color:green"><b>Random Images from the Training Dataset</b></span> code cell contains the `display_random_images()` function which produces twenty random images from the training dataset as shown below:

![alt text][image1]

The <span style="color:green"><b>Histogram of Training Dataset</b></span> code cell produces a histogram of the training dataset as show below (it is evident that the dataset is not balanced):

![alt text][image1b]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sections: <span style="color:orange"><b>Set Datasets to Grayscale</b></span> and <span style="color:orange"><b>Normalize the Datasets</b></span>.

I converted the images to grayscale because color did not add much to the meaning of the signs as they shared similar colors, grayscale may reduce training time and it's a lot easier for the model to classify images if it ignores color.

![alt text][image2]

As a last step, I normalized the image data because ...

As we want the values involved in the calculation of the loss function to never get too big or too small, one good guiding principle is that we always want the variables to have zero mean and equal variance whenever possible. We would like the values we compute roughly around a mean of zero and equal variance when we're doing optimization as it makes it a lot easier for the optimizer to do its job.

![alt text][image2b]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The datasets were already split into training, testing and validation sets. This can be viewed in <span style="color:green"><b>Training, Testing and Validation Datasets</b></span>. 

As stated above, <span style="color:green"><b>Basic Summary of the Data Set</b></span> provides details how much data was in each set.

The <span style="color:green"><b>Data Augmentation</b></span> section is where the data was augmented. I augmented the data because many of the classes were grossly under represented in the datasets. Four common ways to augment the data is to alter existing images by means of scaling, translation, warping and brightness. I applied all four to classes with less than an acceptable number of instances until they comprised an acceptable number. I'd like to investigate additional methods of altering the images as well as faster methods.

Here is an example of an original image and an augmented image:

![alt text][image3]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the <span style="color:green"><b>LeNet Architecture</b></span> section.

My final model consisted of the following layers:

| Layer | Description | 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x1 Grayscale image | 
| L1 Convolution | 1x1 stride, VALID padding, 5x5 filter, output 28x28x16 |
| L1 Activation	| A ReLU activation function. |
| L1 Pooling	 | 2x2 kernel, 2x2 stride, outputs 14x14x6 |
| L2 Convolution | 1x1 stride, VALID padding, 5x5 filter, output 10x10x32. |
| L2 Activation | A ReLU activation function. |
| L2 Pooling	| 2x2 kernel, 2x2 stride, outputs 5x5x32 |
| L2 Flatten	| output 800 |
| L3 Fully Connected |	output 516 |
| L3 Activation | A ReLU activation function. |
| L3 Dropout | 516 |
| L4 Fully Connected | output 84 |
| L4 Activation |A ReLU activation function. |
| L4 Dropout | 360 |
| L5 Fully Connected | 43 |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is trained in the <span style="color:green"><b>Train the Model</b></span> section.

| Parameter | Value | 
|:---------------------:|:---------------------------------------------:|
| EPOCHS | 10 |
| BATCH_SIZE | 256 |
| mu | 0 |
| sigma | 0.1 |
| rate | 0.001 |
| optimizer | AdamOptimizer |

The code for training the model is located in several cells spanning from <span style="color:green"><b>LeNet Architecture</b></span> to <span style="color:green"><b>Train the Model</b></span> of the ipython notebook. 

To train the model, I modified the LeNet architecture from the lesson.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the <span style="color:green"><b>Results on the Training Dataset</b></span>, <span style="color:green"><b>Results on the Test Dataset</b></span> and <span style="color:green"><b>Results on the Validation Dataset</b></span> cells.

| Dataset | Accuracy | 
|:---------------------:|:---------------------------------------------:|
| Training | 0.990 |
| Test | 0.949 |
| Validation | 0.973 |

I started with the LeNet architecture from the lesson because it worked well for the lab.

The LeNet architecture from the lesson did not result in a 0.93 validation accuracy so I began tweaking parameters and architecture. A large gain was provided by using dropout.

I didn't keep up with my note taking of results during the process and once I surpassed the 0.93 accuracy, which I don't believe was an original requirement, I locked-down the experimentation to finish the project and turn it in.

I will go back and work with this further and develop a more rigorous procedure.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the number in the circle is noisy.

The third and fourth images might be difficult to classify because they are skewed.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on new images is in the <span style="color:green"><b>Softmax Predictions</b></span> section of the Ipython notebook.

Here are the results of the prediction:

| Probability | Prediction Image	| Actual Image |
|:-----------------:|:----------------:|:-----------------------:| 
| .17 | Speed limit (50km/h) | Keep right |
| .69 | Right-of-way at the next intersection | Right-of-way at the next intersection |
| .81 | Priority road	 | Priority road |
| 1.0 | Yield | Yield |
| 1.0	| No entry | No entry |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares reasonably to the accuracy on the test set of 0.949, which were done in the <span style="color:green"><b>Results on the Test Dataset</b></span> section of the notebook.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on new images is in the <span style="color:green"><b>Softmax Predictions</b></span> section of the Ipython notebook.

The top five soft max probabilities were:

| Actual Image | SM1 | SM2 | SM3 | SM4 | SM5 |
|:------:|:----:|:----------:|:----------:|:--------:|:---------:|
| Speed limit (50km/h) | Keep right | Speed limit (20km/h) | Dangerous curve to the right | Turn left ahead | Speed limit (60km/h) |
| Right-of-way at the next intersection | Right-of-way at the next | Beware of ice/snow intersection | Double curve | Slippery road | Road work |
| Priority road | Priority road | Stop | No passing | Roundabout mandatory | Vehicles over 3.5 metric tons prohibited |
| Yield | Yield | Stop | No vehicles | Turn right ahead | Speed limit (60km/h) |
| No entry | No entry | No passing | No passing for vehicles over 3.5 metric tons | Vehicles over 3.5 metric tons prohibited | Go straight or left |

A screenshot of the results:

![alt text][image9]

