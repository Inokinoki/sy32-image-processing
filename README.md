# SY32 Image Processing Course Project

This project contains:

- course exercises([TD](/TD))
- homework([TP](/TP))
- final project([facial-detection](/facial-detection))

which are attached to my SY32 Image Processing course.

# Facial Detection

Train a Machine Learning Model to detect faces in images, Support Vector Machine(SVM) as core algorithm, Histogram of Oriented Gradients as feature.

# How to use

1. Download the pre-trained model from project release, put it into the facial-detection folder. You can also train your own using `train.py`.
2. Install dependencies:

```
pip install sklearn sklearn-image numpy
```

3. Unzip `project_test.zip` (and `project_train.zip` if you'd like to train/optimize the model)
4. Run:

```
python test.py
```

or

```
python test.py 1
```

if you only want to test with the `0001.jpg`.

Then you'll get the face description file. There should be position and size info.

And if you choose to detect a single image, the faces in it will be cropped into several files.
