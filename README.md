# Concentration Index Generator

This program is developed based on a concentration index model found in ([Sharma, P. et al., 2019. Student Engagement Detection Using Emotion Analysis, Eye Tracking and Head Movement with Machine Learning](https://arxiv.org/pdf/1909.12913.pdf)). The model uses the computer webcam to calculate a CI based on a composite score of emotional state and eye/head movement, and displays a real-time Engagement and Emotional Classification.

The program follows the emotion weights of the original model but the authors have constructed a distraction index based on eye gaze and eye size. Please see the (Concentration Index.xlsx)for more information. This program does not account for head position, unlike Reddy et. al.

The model for emotional detection is based on:

[Arriaga, O. et. al. "Real-time Convolutional Neural Networks for Emotion and Gender Classification." arXiv preprint arXiv:1710.07557 (2017).](https://arxiv.org/abs/1710.07557)

The code (66% accuracy for emotional detection- as benchmarked by Arriaga et. al.) is adapted from:

## Emotional Detection

> https://github.com/balram2697/Face-Emotion-Recognition

> https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Eye Gaze Detection/Measurement

> https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv-p-1/

## Additional References:

> Reddy, T.K. et. al, 2019. Autoencoding Convolutional Representations for Real-Time Eye-Gaze Detection.

> Rodzi, A.h. et. al., 2018. Vision based Eye Closeness Classification for Driver’s Distraction and Drowsiness Using PERCLOS and Support Vector Machines: Comparative Study between RGB and Grayscale Images.


# Setting up the project on your device

Open Anaconda Prompt:

1) conda create -n tarkeez
2) conda activate tarkeez
3) conda install python
4) conda install pip
5) python -m pip install "tensorflow"
6) conda install -c conda-forge dlib
7) pip install opencv-python
8) pip install playsound==1.2.2
9) "Enter the project directory on your PC"

Run the program on your device and see your camera:

10) python run_local_cv.py

Run the program on your device in the background:

11) python run_local_cv.py --run_background True


# Setting up the project on your macOS device

Use this video: https://www.youtube.com/watch?v=BEUU-icPg78 for prepping Anaconda

Installing Tensorflow-MacOS (Tensorflow dependencies)
conda install -c apple tensorflow-deps

Install base TensorFlow:
pip install tensorflow-macos

Install lib:
python3 -m pip install dlib

The rest is the same as windows
