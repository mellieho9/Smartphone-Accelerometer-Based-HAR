# Smartphone-Accelerometer-Based-HAR
A program predicts the type of physical activity from tri-axial smartphone accelerometer data

## Background
This is my response to the final project assignment from HarvardX PH526x's "Using Python for Research" course. Prompt is as follows:
> In this final project, available to Verified learners only, we'll attempt to predict the type of physical activity (e.g., walking, climbing stairs) from tri-axial smartphone accelerometer data. Smartphone accelerometers are very precise, and different physical activities give rise to different patterns of acceleration.

## Input Data
The assignment provides two files:
1. The first file, [train_time_series.csv](https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+2T2021+type@asset+block/train_time_series.csv), contains the raw accelerometer data, which has been collected using the [Beiwe research platform](https://github.com/onnela-lab/beiwe-backend), and it has the following format:
  <code>timestamp, UTC time, accuracy, x, y, z</code>
  x, y, and z correspond to measurements of linear acceleration along each of the three orthogonal axes.
2. The second file, [train_labels.csv](https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+2T2021+type@asset+block/train_labels.csv), contains the activity labels. the activities have been encoded with integers as follows:
  - 1 = standing
  - 2 = walking
  - 3 = stairs down
  - 4 = stairs up
## My Approach
Since the labels in train_labels.csv are only provided for every 10th observation in train_time_series.csv, this implies that each labeled signal is sampled from 10 signals. I combined the 3 axes components in the time series to create a 4th component - the square root of the sum of the axes' squares. 
1. With the above, I converted the training_time_series dataframe of shape (3744,3) into a numpy array of shape (375,10,4)
2. I extracted features from the array using frequency transformation
3. I split those features into training and test sets with an 80:20 ratio. While the training set is used to the train my classifier model, the test set is used to compute the test accuracy. 
4. I randomized the training set to prevent bias and used oversampling with SMOTE on it to prevent activity class imbalance.
