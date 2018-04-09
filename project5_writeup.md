**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car_Not_Car.jpg
[image2]: ./output_images/Hog_Visualization_Example.jpg
[image3]: ./output_images/Normalized_Hog_Features.jpg
[image4]: ./output_images/Hog_Sampling_Method.jpg
[image5]: ./output_images/Heatmap.jpg
[image6]: ./output_images/Heatmap_Labels.jpg
[image7]: ./output_images/Final.jpg
[image8]: ./output_images/Window_Search_&_Heatmaptest1.jpg
[image9]: ./output_images/Window_Search_&_Heatmaptest2.jpg
[image10]: ./output_images/Window_Search_&_Heatmaptest3.jpg
[image11]: ./output_images/Window_Search_&_Heatmaptest4.jpg
[image12]: ./output_images/Window_Search_&_Heatmaptest5.jpg
[image13]: ./output_images/Window_Search_&_Heatmaptest6.jpg
[image14]: ./output_images/Labels_&_Boxestest1.jpg
[image15]: ./output_images/Labels_&_Boxestest2.jpg
[image16]: ./output_images/Labels_&_Boxestest3.jpg
[image17]: ./output_images/Labels_&_Boxestest4.jpg
[image18]: ./output_images/Labels_&_Boxestest5.jpg
[image19]: ./output_images/Labels_&_Boxestest6.jpg


[video1]: ./output_videos/Vehicle_Tracking_Video_No_Heatmap.mp4
[video2]: ./output_videos/Vehicle_Tracking_Video_Heatmap.mp4
[video3]: ./output_videos/Vehicle_Tracking_Video_Heatmap_Smooth.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started the project by importing all of the libraries to be used throughout the entire project in the first code cell of the IPython notebook. Then, I read in all of the `vehicle` and `non-vehicle` images, this happens in the second code cell of my IPython notebook. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The third code cell in the IPython notebook contains all of the user defined functions that were implemented throughout the project.

I then selected a random vehicle image and explored different `hog` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I performed a visualization exercise by extracting raw hog features, normalizing them, and displaying all of the results. This section happens in the 4th code cell of my IPython notebook. This experimentation was solely for the purpose of visualization and to get a feel of how different parameters affect the extraction of hog features.

Here is an example using the `LUV` color space and HOG parameters of `orient = 9`, `pix_per_cell = 8` and `cell_per_block = 2` (Refer to the fourth code cell of my IPython notebook for more parameter values). These are the resulting images from the HOG experimentation:

![alt text][image2]
![alt text][image3]


#### 2. Explain how you settled on your final choice of HOG parameters.

In the IPython code cell number 5 I started to perform HOG feature extractions for the purpose of training my classifier. Here the values I used for the `extract_features` function:

```
color_space = 'LUV' 
orient = 10 
pix_per_cell = 8 
cell_per_block = 2
hog_channel = "ALL" 
spatial_size = (16, 16) 
hist_bins = 16
spatial_feat = True 
hist_feat = True
hog_feat = True 
```

These values were selected after a lot of experimentation. The LUV color space was the color space with the manimun amount of false positives and with the most accuracy detecting vehicles for me. 

I started the project with an "orient" value of 9 and when increasing it seemed to help the detection of false positives since it is trying to match more hog orientations from the training images to the test image windows. Increasing the orientation does increase the code running time, for the exact same reason mentioned before. 

A pix\_per\_cell value of 8 was chosen after, again, a lot of experimentation, it seemed that a square of 8 x 8 pixels wasn't too small to make the calculations take too long but wasn't too big to miss predict the windows.

At some points the windows seemed to get too small and perhaps the reason why some false positives were detected, in that circumstance a cell\_per\_block value of 2 seemed to help the case since now the window would move around in a block of size 2 x 2, this also made the calculation run a little faster. 

Utilizing all of the color channels for the hog feature extraction gave the best performance and results, possibly because of more data being extracted, it did increase the running time, though.  

A spatial size of 16 x 16 and 16 histogram bins also performed well, despite taking less time to train, the performance diminished when using 32 x 32 spatial size and 32 histogram bins.

I also decided to use all three features for this model, the spatial features, the histogram features, and the hog features. That way providing more data to compare the test images against and hopefully getting greater accuracy.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear Support Vector Machine (SVM) starting from line 30 to line 50 of the fifth code cell in my IPython notebook. I used all of the features extracted using the `extract_features` function on the vehicles and nonVehicles images (8792 vehicle photos and 8968 non vehicle photos). First I created a single array with all of the feature values for both classes, then, I fit a StandardScaler to the data so that all predominant data would be normalized. This data is then scaled using the transform function on the scaled data. Then, a new array `y` is created with the labels for the features, assigning the label '1' to all of the features belonging to cars and the label '0' to all of the features belonging to the not cars class. Later, the `train_test_split` function from the sklearn library es used to split the data into 80% training data and 20% test data.

Finally, the SVM classifier is created on line 47 of the 5th code cell of the IPPython notebook with the code `svc = LinearSVC()` and the `X_train` and `y_train` data is fit (trained) on line 50. This classifier yields an SVC test accuracy of 0.9783. Some predictions are made using the svc function `svc.predict()` and the predictions on 20 labels are as follows:

```
My SVC predicts:      [ 1.  0.  1.  0.  1.  1.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  1.  0.  0.  1.]
For these 20 labels:  [ 1.  0.  1.  0.  1.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  1.  0.  0.  1.]

```


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to take the hog sub-sampling approach from the Vehicle Detection and Tracking class. I realized that using the hog sub-sampling window search was a more efficient way to work on this project since it allows you to extract the hog features once and then use them as many times as you need during experimentation. I used the `find_cars` function given in the lesson as a starting point to perform this method. I did edit the function to implement different scales for the search window and edited the `cells_per_step` from '2' to '1' to increase the window overlap when searching an image. This increase of overlap was key to perform the heatmap and apply a threshold high enough to get rid of all (or most of) the false positives. The edited version of the `find_cars` function can be found in the third code cell of my IPython notebook, line 110 to line 196. 

The scales of 1, 2, and 3 were chosen after pure experimentation. They seemed the get the best results. Using anything lower than one significantly increased the code running time and also significantly increased the false negatives, potentially because of comparing features of such small blocks with features of 64 x 64 train images.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Here are some example images:

##### Example of Window Search:
![alt text][image4]

##### Example of Heatmap Implementation:
![alt text][image5]

##### Example of Labels Implementation:
![alt text][image6]

##### Example of Final Thresholded Image:
![alt text][image7]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here I will link three different videos to show the evolution of the pipeline and outputs that came from them:

First, I used the windows found from the `find_cars` function only (with no heatmap and thresholding), this is the resulting video:
[First Video](./output_videos/Vehicle_Tracking_Video_No_Heatmap.mp4)

Then, I used the windows resulted from the heatmap and thresholding, this is the resulting video:
[Second Video](./output_videos/Vehicle_Tracking_Video_Heatmap.mp4)

Finally, I created a new function called `smooth_heatmap` (3rd code cell line 232 to 245) in which a deque kept track of the values of the last 15 heatmaps, calculated the mean and used this result for the heatmap window to display in the picture, this is the resulting video:
[Third Video](./output_videos/Vehicle_Tracking_Video_Heatmap_Smooth.mp4)

It is very visually noticeable that the wobbly aspect of the windows goes away when taking the mean over 15 heatmaps and gives a cleaner output.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that heatmap to identify vehicle positions and get rid of false positives. I then used the `label()` function to identify individual blobs in the heatmap and assumed that each one of them corresponded to a vehicle. Finally, I used the `draw_labeled_bboxes` to draw a box in place of the heatmap blobs. This entire process can be found in code cell number 5 of my IPython notebook, line 24 to 64. It can also be found in code cell 8 of my IPython notebook, line 10 to 23.

Refer to the examples of window search, heatmap implementation, labels implementation, and final thresholded image above for examples of the mentioned implementation. 

### Here are six frames showing sliding window search, their corresponding heatmaps, outputs of `label()` function, and resulting bounding boxes:

![alt text][image8]
![alt text][image14]
![alt text][image9]
![alt text][image15]
![alt text][image10]
![alt text][image16]
![alt text][image11]
![alt text][image17]
![alt text][image12]
![alt text][image18]
![alt text][image13]
![alt text][image19]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For this project I took a Hog Sub-Sampling approach, this approach saved me time when extracting features since the extraction only needs to be done once and all features can be used at any time. The experimentation was very intense, a lot of parameters to tune and therefore many different combinations to try. The heatmaping and thresholding are key in this project, without these, creating a pipeline with no false positives would be a near impossible task. Thresholding is the best way to effectively weed out any false positive detections. The overall process is very long as it takes a lot of time (an patience!). To help this and boost efficiency, I recommend coding some sort of beeping noise at the end of certain code cells so that the beeping will alert you that the code cell is done running (refer to code cell 5, line 64 to 66), and that way be able to focus on different other tasks while the code is running (which, again, is very time consuming) and go on to checking any results once the computer beeps.

An important tip is to limit the y values of the image so that the search for cars happen where cars are actually expected. I used a y start of 400 and a y stop of 720 so that I could focus the search on the lower part of each image instead of wasting time and possibly getting false positives from searching around the upper part of the image (with mostly sky and trees).

Some functions could be improved to take more inputs, for example, my `find_cars` function has the conversion from RGB to LUV hardcoded in when it could be an input parameter to the function. The list of scale values could also be an input parameter instead of hardcoded into the `find_cars` function.

Next, I would like to combine the output pipeline of Project 4 with Project 5 and therefore be able to find lane lines and at the same time detect and track vehicles.

