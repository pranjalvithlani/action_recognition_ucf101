# action_recognition_ucf101
recognices actions from the videos 

## Recognizing Actions in Videos
### 

---

**Action Detection Project**

The goals / steps of this project are the following:

* Dataset prepration and pre-processing
* Selecting the model architecture suitable for videos
* Evaluation

[//]: # (Image References)
[image1]: ./examples/UCF101.jpg
[image2]: ./examples/abc.jpg
[image3]: ./examples/(2+1)D_vs_3D.jpg
[image4]: ./examples/eyemakeup+lipstick.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

### Dataset prepration and pre-processing

#### 1. UCF101 Dataset

The UCF101 dataset is pretty famous and mostly used by all the researchers when performing any task with videos. The link to the dataset is [Here](https://www.crcv.ucf.edu/research/data-sets/ucf101/). It has 13320 videos from 101 action categories and UCF101 gives the large diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc. The videos in 101 action categories are grouped into 25 groups, where each group can consist of 4-7 videos of an action. The videos from the same group may share some common features, such as similar background, similar viewpoint, etc.

The action categories can be divided into five types:

* Human-Object Interaction
* Body-Motion Only
* Human-Human Interaction
* Playing Musical Instruments
* Sports

![alt text][image1]



#### 2. Frames Extraction

For the modeling, Each video clip is represented by 8-uniformaly sampled frames from that clip. So, rather than using single image to represent a video, using sequence of frames introduces temporal information into the context. 

frame example
![alt text][image2]


### Selecting the model architecture suitable for videos

#### 1. Related work

The Video understanding is one of the core computer vision problems and has been studied for decades. Some proposed video representations include spatiotemporal interest points eg. SIFT-3D and HOG3D. These representations are hand-designed and use different feature encoding schemes such as those based on histograms or pyramids. Then came the deep learning era which leveraged CNNs for most of the CV tasks. Use of CNN for video understanding is done in several ways:- 2D convolutions over the entire clip, 3D convolutions, C3D, I3D, Mixed Convolutions(2D + 3D, or 3D + 2D), CNNs + RNNs, etc. 

#### 2. Model Architecture used

Here, I'm using (2+1)D ConvLayer in place of 3D. It has been providing the state-of-the-art accuracy in so many video understanding tasks. I'm using a popular ResNet, 18 layer architecture for training the model. To get the benefits of transfer learning, pre-trained ResNet weights on Kinetics-400 dataset is used. 

![alt text][image3]

In this image above, a) is a Full 3D Convolution and  (b) A (2+1)D convolutional block splits the computation into a spatial 2D convolution followed by a temporal 1D convolution.

### Evaluation

#### 1. Model Accuracy

##### After train the model for several epochs using SGD with momentum, a State-Of-The-Art accuracy of ~98% is achieved. 

#### 2. Hard Testing

Testing our model on real movies available on internet. I've take a clip the Indian movie (Manmadhudu). The extracted frames are:-

![alt text][image4]

Here the data is ambiguious as the actor is having lipstick in his hand and is applying it on his eyes, rather than on his lips. So, our model predicts it as an APPLYING LIPSTICK category. But intersting thing is, if we check the top-5 predictions:- 
1) ApplyLipstick
2) Brushing Teeth
3) ApplyEyeMakeup
4) ShavingBeard
5) Haircut

The good thing about model is, it can predict APPLY EYE MAKEUP in its top-5 predictions, even after the video is confusing. This shows how well the model can generalize to the movie data and can recognise actions from the movies as well.
