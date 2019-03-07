import numpy as np
import torch
from torch import nn, optim, utils, cuda, device
from torchvision import datasets, transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
from workspace_utils import active_session
import time
import argparse

def preprocess_data():
    train_data_transforms = None
    validation_data_transforms = None
    test_data_transforms = None
    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                                           std = [ 0.229, 0.224, 0.225 ]),
                                                ])

    validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                                    std = [ 0.229, 0.224, 0.225 ]),
                                              ])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                                    std = [ 0.229, 0.224, 0.225 ]),
                                              ])
    return train_data_transforms, validation_data_transforms, test_data_transforms

# TODO: Load the datasets with ImageFolder
def create_load_data(train_dir, valid_dir, test_dir, train_data_transforms,validation_data_transforms,test_data_transforms,batch_size=64):
    # TODO: Load the datasets with ImageFolder
    trainloader = None
    validationloader = None
    testloader = None
    
    train_dataset = datasets.ImageFolder(train_dir,transform=train_data_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir,transform=validation_data_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    validationloader = utils.data.DataLoader(validation_dataset,batch_size=batch_size,shuffle=True)
    testloader = utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    
    return trainloader, validationloader, testloader

# TODO: Build and train your network
#load model using model string specified by user in command line script.
def load_model(model_type):
    model = None
    model_dict = {'vgg16':'models.vgg16(pretrained=True)',
                  'alexnet':'models.alexnet(pretrained=True)',
                  'densenet':'models.densenet161(pretrained=True)'}
    try:
        model = eval(model_dict[model_type])
    except Exception as e:
        print('Likely an error in model type.\nExpected model types are\n \
                1. \'vgg16\'\n \
                2. \'alexnet\'\n \
                3. \'densenet\'\n \
               ')
        print('Entered model type -- {}'.format(model_type))
    return model

#Based on type of pre-trained model, return the number of input features required at the classifier input.
#This will be invoked in the custom classifier built below.
def model_classifier_out_features(model):
    key = str(type(model)).split('\'>')[0].split('.')[-1]
    typedict = {'AlexNet':'model.classifier[1].in_features',
                'VGG':'model.classifier[0].in_features',
                'DenseNet':'model.classifier.in_features'
                }
    return eval(typedict[key])

class CustomNetwork(nn.Module):
    def __init__(self,num_hidden_layers=2,n_features_first_layer=25088,n_features_hidden_layers=4096, n_features_output=102):
        super(CustomNetwork,self).__init__()
        self.num_hidden_layers=num_hidden_layers
        self.n_features_first_layer = n_features_first_layer #n_features_first layer must be passed to be equal to the number of outputs in the template
        self.n_features_hidden_layers = n_features_hidden_layers
        self.n_features_output = n_features_output
        self.h0 = nn.Linear(in_features=self.n_features_first_layer, \
                            out_features=self.n_features_hidden_layers, bias=True)
        self.r0 = nn.ReLU()
        self.d0 = nn.Dropout(p=0.5)
        for h in range(1,self.num_hidden_layers-1):
            
            # Inputs to hidden layer linear transformation
            #NOTE: The network as defined here is a little limited in the sense that a user cannot specify\
            #the node structure and parameters (dropout, activation) in each hidden layer.\
            #This is not required in the rubric, but seems like something\
            #that could/should be implemented
            setattr(self, 'h'+str(h+1), nn.Linear(in_features=self.n_features_hidden_layers, out_features=self.n_features_hidden_layers, bias=True))
            setattr(self, 'r'+str(h+1), nn.ReLU())
            setattr(self, 'd'+str(h+1), nn.Dropout(p=0.5))

        #Output layer
        self.ho = nn.Linear(in_features=self.n_features_hidden_layers, out_features=self.n_features_output, bias=True)
        self.output = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.h0(x)
        x = self.r0(x)
        x = self.d0(x)
        for h in range(1,self.num_hidden_layers-1):
            x = getattr(self,'h'+str(h+1))(x)
            x = getattr(self,'r'+str(h+1))(x)
            x = getattr(self,'d'+str(h+1))(x)
        x = self.ho(x)
        x = self.output(x)

        return x
#train fresh model or retrain existing model
def train_model(model,criterion=nn.NLLLoss(),epochs=1,learning_rate=0.001):
    with active_session():
        #explicitly setting this so that the same function can be used for training and retraining
        #First freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        #Unfreeze model classifier parameters
        for param in model.classifier.parameters():
            param.requires_grad=True
            
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
        #Convert to GPU or CPU as appropriate
        model.to(device)
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for train_inputs, train_labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

                optimizer.zero_grad()

                train_log_probabilities = model.forward(train_inputs)
                train_loss = criterion(train_log_probabilities, train_labels)
                train_loss.backward()
                optimizer.step()

                running_loss += train_loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for validation_inputs, validation_labels in validationloader:
                            validation_inputs, validation_labels = validation_inputs.to(device), validation_labels.to(device)
                            validation_log_probabilities = model.forward(validation_inputs)
                            batch_loss = criterion(validation_log_probabilities, validation_labels)

                            validation_loss += batch_loss.item()

                            # Calculate accuracy
                            validation_probabilities = torch.exp(validation_log_probabilities)
                            top_p, top_class = validation_probabilities.topk(1, dim=1)
                            equals = top_class == validation_labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()
    return model

# TODO: Do validation on the test set
def measure_holdout_performance(model,testloader,criterion=nn.NLLLoss()):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

if __name__ == '__main__':
    #Check to see if on GPU or CPU
    gpu_avail = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_avail else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--train",required = False,action='store_true',
                        help="If flag is used, train model from scratch. Else supply saved model using -m/--model <MODEL_FILEPATH> option")
    parser.add_argument("-m","--model",required=False,default='ImageClassifier.model',
                        help="Path to file in which classifier model is stored. Used only if -t/--train is not set explicitly. Ignored otherwise.")
    parser.add_argument("-lr","--learning_rate",required=False, type=float,
                       default=0.001,help="Learning rate for optimizer")
    parser.add_argument("-e","--epochs",required=False, type=int,
                       default=1,help="Number of epochs to train over.")
    parser.add_argument("-u","--units_hidden",required=False,type=int,
                       default=2,help="Number of hidden units in the custom classifier")
    parser.add_argument("-n","--nodes",required=False,type=int,
                       default=4096,help="Number of nodes per hidden layer in the custom classifier")
    parser.add_argument("-s","--savepath",required=False,default='ImageClassifier.model',
                       help="Path at which trained/retrained model can be saved. If not mentioned, model is saved in default path \'ImageClassifier.model\'")
    parser.add_argument("-mt","--model_template_type",required=False,default='vgg16',
                       help="Pre-trained models available in Torchvision to be used as template to build classifier. \n\
                       Used only when -t/--train flag is set. Ignored otherwise.\
                       \nChoose from:' \n \
                       1. \'vgg16\'\n \
                       2. \'alexnet\'\n \
                       3. \'densenet\'\n \
                       ")
    parser.add_argument("-g","--gpu",required=False,action='store_true',
                        help="Entering flag allows use of GPU to train when available.")
    args = parser.parse_args()

    #Parse arguments and set
    train = args.train
    model_path = args.model
    lr = args.learning_rate
    epochs = args.epochs
    custom_classifier_hidden_layers=args.units_hidden
    custom_classifier_nodes_per_layer=args.nodes
    use_gpu = args.gpu
    savepath = args.savepath
    model_template_type=args.model_template_type

    #In Principle, directories can also be input from user, but this is not required in rubric, so I am not putting it in.
    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Transform and load test, validation and train datasets
    train_data_transforms, validation_data_transforms, test_data_transforms = preprocess_data()
    trainloader, validationloader, testloader = create_load_data(train_dir, valid_dir, test_dir, train_data_transforms,validation_data_transforms,test_data_transforms)

    #Load Model
    if(train==True):
        model = load_model(model_template_type) #load template type specified in script. If not specified load default - 'vgg16'
        model_template = load_model(model_template_type) #loading another copy of model template since my custom Network needs this information
        custom_clf_in_features=model_classifier_out_features(model_template)
        custom_clf_n_hidden = custom_classifier_hidden_layers
        custom_clf_nodes_per_hidden = custom_classifier_nodes_per_layer
        model.classifier = CustomNetwork(num_hidden_layers=custom_clf_n_hidden,\
                                         n_features_first_layer=custom_clf_in_features,\
                                         n_features_hidden_layers=custom_clf_nodes_per_hidden,\
                                         n_features_output=102) 
        print('Training...\n')
    else:
        if(os.path.exists(model_path)):
            try:
                model = torch.load(model_path,map_location=lambda storage, loc: storage)
                print('Continuing to train existing model...\n')
            except Exception as e:
                print(e)
        else:
            print('Enter model file if -t/--train flag is not entered. Either\n\
                   1. Model File does not exist at entered path or \n\
                   2. Model not present at default path \'ImageClassifier.model\'')



    model = train_model(model,epochs=epochs,learning_rate=lr)
    measure_holdout_performance(model,testloader)
    # TODO: Save the checkpoint 
    train_dataset = datasets.ImageFolder(train_dir,transform=train_data_transforms)
    model.class_to_idx = train_dataset.class_to_idx
    torch.save(model, savepath)
    #torch.save(model.state_dict(), './VGG_2L_4096N.statedict') # Preferred way in Pytorch but still figuring out how to get this to work