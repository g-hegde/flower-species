# flower-species

This project was my submission for the Deep-learning project requirement of Udacity's Data Science Nanodegree program.
The objective here is to create a flower-species classifier using Transfer Learning in PyTorch.
The dataset was provided by Udacity. Separate train/, test/ and validate/ folders contained class folders (named 1/ through 102/) which in turn contained .jpeg images of flowers. Convolutional Neural Networks (CNN) pre-trained on ImageNet and available in PyTorch were used for feature creation. A custom classifier with user-programmable hidden layers and nodes was then trained on the image dataset.

image-classifier.ipynb contains the classifier code that eventually found it's way into two command line applications (train.py and test.py).
train.py contains functions and classes for command line based user input and training of the custom classifier.
test.py uses models trained in train.py to predict the top-k classes and their associated classification probabilities and class names.

Files included:
1. train.py - Train a Custom Neural Network classifier that sits on top of a template used as a feature detector.
2. predict.py - Predict flower classes using a trained model.
3. workspace_utils.py - Udacity supplied script to keep session running even when user is inactive.
4. cat_to_name.json - Map flower class/category to name
5. Image Classifier Project.ipynb - Notebook for part 1.
6. README.md - this file.

Files/Directories needed for correct operation:
1. flowers/ - Needs to contain train/ test/ and validate/ directories with images belonging to each category in each directory.

Example Usage:

train.py
1. python3 train.py --train
Trains a Custom Neural Network on top of VGG16 with 2 hidden layers and 4096 nodes per hidden layer. Saves the resulting model in 'ImageClassifier.model'
2. python3 train.py --train --gpu
Same as 1. above, except that it uses the GPU for training if available
3. python3 train.py --train --gpu --units_hidden 3 --nodes 1024 -lr 0.001 --e 4 --savepath 'ImageClassifier.model' -mt 'alexnet'
Train a Custom Neural Network Classifier on top of a AlexNet backbone and save the resulting model to 'ImageClassifier.model'. The classifier contains 3 hidden layers, 1024 units per hidden layer and is trained for 4 epochs using the Adam optimizer and a learning rate of 0.001.
4. python3 train.py -m "ImageClassifier_alexnet.model" -s "ImageClassifier_alexnet.model"
Retrain model for a default of one epoch using a default learning rate of 0.001 and save back to same model

predict.py
1. python3 predict.py -f "flowers/test/37/image_03789.jpg" -m "ImageClassifier_alexnet.model" --gpu -n
Predict top 5 classes, probabilities, class names for picture using model on gpu
2. python3 predict.py -f "flowers/test/37/image_03789.jpg" -m "ImageClassifier_alexnet.model" --gpu -n -t 20
Same as 1. above, except that JSON file will be loaded and top 20 classes, probabilities and names will be printed.


