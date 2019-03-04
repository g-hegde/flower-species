# flower-species

This project was my submission for the Deep-learning project requirement of Udacity's Data Science Nanodegree program.
The objective here is to create a flower-species classifier using Transfer Learning in PyTorch.
The dataset was provided by Udacity. Separate train/, test/ and validate/ folders contained class folders (named 1/ through 102/) which in turn contained .jpeg images of flowers. Convolutional Neural Networks (CNN) pre-trained on ImageNet and available in PyTorch were used for feature creation. A custom classifier with user-programmable hidden layers and nodes was then trained on the image dataset.

image-classifier.ipynb contains the classifier code that eventually found it's way into two command line applications (train.py and test.py).
train.py contains functions and classes for command line based user input and training of the custom classifier.
test.py uses models trained in train.py to predict the top-k classes and their associated classification probabilities and class names.


