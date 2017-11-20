# behavioral-cloning-deep-learning
Train a simulated car to drive using behavioral cloning in keras:

[![Behavioral cloning](https://img.youtube.com/vi/ZhDZ16ReC_8/0.jpg)](https://www.youtube.com/watch?v=ZhDZ16ReC_8 "Behavioral cloning")

## Keywords: Behavioral Cloning, Keras, Deep learning, autonomous car

The goals of this project are the following:
* Use a simulator (provided by Udacity) to collect data of good driving behavior. "Good" driving is done manually using the keyboard/mouse
* Build a convolution neural network in Keras that predicts steering angles from the training images obtained from above
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

This repo includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

The model.py file is the starting point for loading data and training the network. Other tasks are
handled in separate files, which are included in the submission. The files show the pipeline I used
for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Model architecture
My model is derived from the nVidia self-driving car model which is implemented as nVidiaNet in
p3_model.py. This model consists of a convolution neural network with 3x3 filter sizes and
depths between 16 and 64.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the
model during the pre-processing stage

#### Attempts to reduce overfitting in the model
The model contains dropout layers to reduce overfitting. Dropout is set to a fixed value of 0.5
through the training.
The model was trained and validated on different data sets to ensure that the model was not
overfitting. The model was tested by running it through the simulator and ensuring that the
vehicle could stay on the track.

#### Model parameter tuning
The model used an adam optimizer. Additionally, a callback was used for the keras fit generator
that reduces the learning rate if the validation loss increased monotonically for a threshold
number of epochs. Further, the model was saved after every epoch so that each epoch could be
tested on the simulator. It was noted that validation loss was not an accurate estimator of
driving quality.

#### Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of center
lane driving, recovering from the left and right sides of the road

#### Data collection strategy
To collect data, the simulator was used. The following data was recorded:
* Driving forward on the track - 2 laps
* Driving forward on the reverse track - 2 laps
* Extended data on the bridge
* Recovery data as required
  Once the data above was collected and the network was trained, the simulator
  was run on autonomous mode. If the car crossed the lanes at some point, more
  recovery data was recorded for that part of the track. For recovering from the left
  lane, the car was driven from the shoulder to the center and vice-versa for
  recovery from the right lane.
  
The mouse was used to steer the car while recording the data since data obtained by using the
keyboard was not smooth and is very choppy:

![Steering angles obtained](/images/keybrd_mouse.png)

As can be seen in the picture above (driving forward on the track), using the mouse gives a more
continuous steering angle label and is beneficial to training of the network. If the keyboard data
is used, there are too many falsely labelled images with 0 steering angle which makes the
network try and guess a constant value to minimize error during training.

#### Pre-processing the data
While reading data, and creating the master CSV list, all data was cropped from (160x320) to
(80x320); to remove irrelevant parts of the image that do not contribute to driving behavior. The
top and bottom of each image were cropped. Further, this image to was resized to (40,160) to
reduce the load on the network while training.

![Original and final image after pre-processing](/images/orig_cropped.png)

### Model Architecture and Training Strategy
#### Solution Design Approach
The overall strategy for deriving a model architecture was to:
* Collect data from the simulator. How much data to collect was not clearly known at this
point
* Start with the network used in the last project (traffic sign classification), and slowly increase complexity if needed

However, the network used in the previous project was not very good at learning the data. I then
decided to use the nVidia model architecture since this has been proven to work for the same
use case before.
To gauge how well the model was working, I split my image and steering angle data into a
training (80%) and validation (20%) set. A validation loss was calculated as the mean absolute
error between the true and predicted steering angles for the validation set and this was used to
monitor progress of training.

To combat overfitting, dropout was used after each of the fully connected layers and a dropout
probability of 50% was used throughout the training.

The model was trained for 20 epochs with a learning rate of 1e-4.

The final step was to run the simulator to see how well the car was driving around the track.
There were a few spots where the vehicle fell off the track. This was around sharp turns, before
and after the bridge etc. There were also areas where the car hugged either lanes before
recovering to the center. To improve the driving behavior in these cases, I recorded additional
recovery data at these spots and added this to the training set and re-run the training.
At the end of the process, the vehicle can drive autonomously around the track without leaving
the road.

#### Final Model Architecture
The final model architecture consisted of a convolution neural network with the following layers
and layer sizes:
Input: (40x160x3)
1. Layer 1: Convolutional 2D. Filter size (3x3), depth 32
2. Activation: ReLU
3. Layer 2: MaxPooling2D. Pool size (2x2), strides (2x2)
4. Layer 3: Convolutional 2D. Filter size (3x3), depth 64
5. Activation: ReLU
6. Layer 4: MaxPooling2D. Pool size (2x2), strides (2x2)
7. Layer 5: Convolutional 2D. Filter size (3x3), depth 128
8. Activation: ReLU
9. Layer 6: MaxPooling2D. Pool size (2x2), strides (2x2)
10. Layer 7: Flatten image
11. Layer 8: Fully Connected Layer. Output: 500
12. Activation: ReLU
13. Layer 9: Dropout @ 50%
14. Layer 10: Fully Connected Layer. Output: 100
15. Activation: ReLU
16. Layer 11: Dropout @ 50%
17. Layer 12: Fully Connected Layer. Output: 20
18. Activation: ReLU
19. Layer 13: Dropout @ 50%
20. Layer 14: Fully Connected Layer. Output: 1
The biases in the above architecture are initialized to 0 and the weights are initialized using the
glorot_normal initializer in keras. The model contains a total of 3,601,889 trainable parameters!

#### Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving
in forward and reverse directions. Here is an example image of center lane driving:

![Center lane driving](/images/center_lane_driving.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to
center so that the vehicle would learn to remain in the center and not veer off course. These
images show what a recovery looks like:

![Recovery from the left lane](/images/recov_left_01.jpg)
![Recovery from the left lane](/images/recov_left_02.jpg)

![Recovery from the right lane](/images/recov_right_01.jpg)
![Recovery from the right lane](/images/recov_right_02.jpg)

About 35,000 images were gathered as part of the step above. This data is augmented to:
* Increase the size of the dataset and help the network generalize
* Present slightly perturbed data to the network so it generalizes better on the track
The following two data augmentation techniques were used:

**Adding random shadows**
Boxes of random sizes are drawn on the original image and filled with varying shades of grey. The
shadows are chosen at random for this. This doubles the size of the dataset.
**Changing the brightness of the image**
The brightness of the training images was randomly scaled by converting to the YUV color space
and modifying the V channel. This was done for each of the images in the original dataset.
The image below shows what the augmented images look like:

![Augmented image](/images/augmented_images.png)

At the end of the above augmentations, I had 3x the original number of images.
A generator was used to augment the data on the fly such that at any time, only 129 images
were loaded in memory and passed to the fit generator in keras for both training and validation.
This significantly reduced the memory usage on my PC.
I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model
was over or under fitting. The ideal number of epochs was ~20 as evidenced by the value of the
validation loss. However, as mentioned earlier, the validation loss is not a very good indicator of
driving quality. Therefore, the epochs with the 5 lowest validation losses were tested each time
the network was trained.

#### Challenges and areas of improvement
The following challenges were faced and overcome during this project:
1. Network predicting constant steering angles:
During the initial phase of training, the network would produce a constant prediction for steering
angle regardless of the input. This was found out to be due to an unbalanced dataset with more
entries for zero steering angle than others.
2. Too little vs. too much data:
When this model was trained with too little (<20k images) or too many (>100k) training data samples, it tended to overfit and would go off lane on track 1. The sweet spot of right amount of data was found through trial and error.
The current code can steer the car such that it can drive autonomously over the track. However, there are
several improvements that can be made:
1. Keep the car from going over either lane markings. 
This may be accomplished by more recovery data and/or a larger network
2. Teaching the vehicle to steer and accelerate/brake autonomously. 
Since there are no obstacles/traffic on this test track, it can be used to investigate how the throttle may also be
autonomously controlled.
