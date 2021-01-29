# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/av_model.png "Model Visualization"
[image2]: ./examples/center_driving_example.jpg "Center Driving Image"
[image3]: ./examples/bridge_practice_run.jpg "Bridge Driving Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The final model was trained by invoking the following command
```
python model.py --basedir train_data_new --epochs 3 --augmentations 10000
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a convolutional neural network with 5 Conv2D layers with kernels of size 5x5 (for the first 3 conv layers) and 3x3 (for the last 2 conv layers).
Each conv layers is followed by a Dropout layer and the first 3 conv layers subsample (with stride 2x2).
The conv layers are followed up by 4 Dense layers of progressively decreasing size (from 100 to 1).

The complete model is implemented in the function get_nvidia_model() in models.py file.

The model includes RELU activations after each conv layer to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer and the images are cropped from the top and bottom to increase training speed and remove unnecessary info (i.e. the trees and sky at the top and the hood of the car at the bottom of the image). 

Here's the model summary.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

My initial attempts to train the model resulted in early overfitting with the training loss was steadily going down while the validation loss was going up (and always much higher than the training loss).
In order to reduce overfitting I added Dropout layers and after a few iterations, I got best results by placing dropout layers after every conv layer (with drop prob set at 0.5).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and passing the bridge several times.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start small and simple and build up gradually adding more complexity.
Thus, I first made sure that the simulation and training framework were working as expected by training the most basic model (get_basic_model() in models.py).
This model could drive the car but the car was unable to stay on the road for more than a few seconds.

Next I added the normalization layer and re-trained the model. Again, the results weren't much better than the first iteration.

Then I introduced the first conv net by implementing LeNet model (see get_lenet_model() in models.py).
This model was clearly doing something well as it was able to drive the car through a couple of turns but would inevitably run off the track just the same.
This was when I started to collect more data. First I drove exactly one lap clockwise and one lap counterclockwise around the track.
Here I also inroduced the cropping layer to speed up training and reduce unnecessary info processed by the model.
After training the model I started to pay close attention to how the car behaved and added increasingly more training data.
First, I practiced many recovery maneuvers by starting at the left/right side of the track and re-centering the car.
After a few iterations I noticed that the model would consistently struggle getting on and off the bridge or passing it smoothly. So I took my training runs in the simulator multiple times across the bridge and practiced recentering on the bridge and approaching to the bridge.
This seemed to solve the bridge problem but the car was still getting stuck at some spots on the track.
While collecting more data, I noticed that my training runs were getting very slow and waiting for them was taking too much time. So I added an extra flag to my training pipeline file `--agumentations` that would allow me to set the max number of images to augment (by flipping them horizontally and adding left and right camera views).
This helped to curb the training time but kept the model performance on par with the more expensive fully augmented training data runs.

After realizing that LeNet wouldn't solve the problem any time soon, I implemented the more complex Nvidia model that had additional conv/dense layers.
Although the Nvidia model showed immediate improvement over the LeNet, I was still struggling to keep the car on track for a whole lap.
I tried to collect more data by focusing on the problematic spots but this didn't resolve the issue and instead became a game of Whac-A-Mole - each time I'd resolve one problematic spot, something else would arise along the way.
Then I noticed that my model was consistently overfitting and I realized I needed to regularize it.
Since adding more and more data wasn't helping the overritting issue (not noticeably at least), I tried adding Dropout layers after the first two conv layers.
This helped with the overfitting but the car was still not perfect in its driving. Then I experimented by inserting Dropout layers after each of the conv layer and finally put Dropout layer after every conv layer.
This clearly addressed the bulk of my overfitting issues as the training and validation loss were tracking equally throughout a 10 epoch training run.
However, the car was still having minor issues that would eventually lead it off track. As the final touch I decided to stop training after the 3rd epoch (since the validation loss reached its minimum there) and voila!
When I ran this retrained model in autonomous mode in the simulator, the car was able to successfully go around the entire track smoothly.
Here's the training run loss for the last model (the final model training was stopped after the 3rd iteration):
```
38597/38597 [==============================] - 66s 2ms/step - loss: 0.0498 - val_loss: 0.0327
Epoch 2/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0458 - val_loss: 0.0329
Epoch 3/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0446 - val_loss: 0.0322
Epoch 4/10
38597/38597 [==============================] - 61s 2ms/step - loss: 0.0437 - val_loss: 0.0391
Epoch 5/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0434 - val_loss: 0.0368
Epoch 6/10
38597/38597 [==============================] - 61s 2ms/step - loss: 0.0427 - val_loss: 0.0339
Epoch 7/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0424 - val_loss: 0.0347
Epoch 8/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0419 - val_loss: 0.0347
Epoch 9/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0418 - val_loss: 0.0350
Epoch 10/10
38597/38597 [==============================] - 62s 2ms/step - loss: 0.0414 - val_loss: 0.0335
```
#### 2. Final Model Architecture

The final CNN model architecture (models.py lines 34-51) consisted of a normalization layer, a cropping layer, 5 conv layers (followed by ReLU activation and dropout) and 4 fully connected layers.
Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on the track one using center lane driving (1 lap clockwise and 1 lap counterclockwise). Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to not get stuck on the road side and able to recenter itself.

I also did a number of practice runs through the bridge, since it was initially a tough segment for the car to traverse.

![alt text][image3]

To augment the data set, I also flipped images horizontally to make left and right turns more balanced. I also used images from both left and right cameras (in addition to the center camera).
I only applied augmentation to the first 10k images to avoid excessively long data pre-processing times.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the lowest validation loss achieved at the end of each epoch in a 10 epoch training run.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
