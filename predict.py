# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:59:51 2019

@author: Ganesh
"""
# Imports here
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse
from train import CustomNetwork

gpu_avail = torch.cuda.is_available()
device = torch.device("cuda" if gpu_avail else "cpu")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
 # TODO:    Process a PIL image for use in a PyTorch model

    processed_image = None
    
    #Resize
    width,height = image.size
    if(height<width):
        AR = float(width)/float(height)
        size = (int(AR*256),256)
    else:
        AR = float(height)/float(width)
        size = (256,int(AR*256))
    image.thumbnail(size) #inplace

    #Crop, keeping in mind that the center coordinates of the resized image have changed
    width,height = image.size
    new_width,new_height=224,224
    zero_w,zero_h = width/2,height/2
    box = (zero_w-new_width/2,zero_h-new_height/2,zero_w+new_width/2,zero_h+new_height/2)
    image = image.crop(box)

    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image-mean)/std

    processed_image = norm_image.transpose((2,1,0))
    return processed_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    image = Image.open(image_path)
    preprocessed_image_tensor = torch.from_numpy(process_image(image)).float().to(device)
    with torch.no_grad():
        model.eval()
        logps = model.forward(preprocessed_image_tensor.unsqueeze_(0))
        ps = torch.exp(logps)
        top_p, top_class_indices = ps.topk(topk, dim=1)
        idx_to_class = dict([value,key] for key,value in model.class_to_idx.items())

        if(gpu_avail==True):
            top_classes = [idx_to_class[idx] for idx in top_class_indices.cpu().numpy()[0]]
            top_p = top_p.cpu()
        else:
            top_classes = [idx_to_class[idx] for idx in top_class_indices.numpy()[0]]
        
    return top_p.numpy()[0],top_classes

if __name__ == '__main__':
    #Let user specify the path to the image file
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file",required = True,
                        help="Path to image file for which class prediction is needed.")
    parser.add_argument("-m","--model",required=True,
                        help="Path to file in which classifier model is stored. ")
    parser.add_argument("-t","--top",required=False, type=int,
                       default=5,help="Display TOP classes, and probabilities")
    parser.add_argument("-n","--name",required=False,action='store_true',
                       help="Entering flag allows loading of JSON file and printing of class names along with classses and probabilities")
    parser.add_argument("-g","--gpu",required=False,action='store_true',
                        help="Entering flag allows use of GPU to predict classes when GPU is available.")
    args = parser.parse_args()

    image_path = args.file
    model_path = args.model
    topk = args.top
    use_gpu = args.gpu
    name = args.name

    print('\nImage path entered -- {}\n'.format(image_path))
    print('Model path entered -- {}\n'.format(model_path))
    print('Top {} classes with class names and probabilities requested\n'.format(topk))
    print('Use GPU to predict classes? {}\n'.format("Yes" if use_gpu else "No"))
    print("GPU Available? {}\n".format("Yes" if gpu_avail else "No"))
    if(os.path.exists(image_path) and os.path.exists(model_path)):
        model = torch.load(model_path,map_location=lambda storage, loc: storage)
        model.to(device)
        if((gpu_avail != True) and (use_gpu == True)):
            print('GPU not available. Using CPU for prediction...\n')
        probs,classes = predict(image_path, model,topk)
        print('Most Probable Image class is {} with probability {}\n'.format(classes[0],probs[0]))
        if(name==True):
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
                class_names = [cat_to_name[cat] for cat in classes]
                class_dict = {'Class Name':class_names,'Class':classes,'Probability':probs}
                print('Displaying Top {} Classes, Class Names and Probabilities\n'.format(topk))
        else:
                class_dict = {'Class':classes,'Probability':probs}
                print('Displaying Top {} Classes and Probabilities\n'.format(topk))
        class_df = pd.DataFrame(class_dict)
        print(class_df.to_string(index=False))
        print("\n\n")
    else:
        print('Model file path and/or image file path incorrect. Exiting program..\n')
