# Imports here
import pandas as pd
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import models,datasets,transforms
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
import argparse
import json


app = argparse.ArgumentParser(
    description='predict.py')
app.add_argument('--input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06754.jpg', dest="input_img", action="store", type = str)
app.add_argument('--checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth',dest="checkpoint", action="store",type = str)
app.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
app.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
app.add_argument('--gpu', default="cuda", action="store", dest="gpu")

pars = app.parse_args()
path_image = pars.input_img
number_of_outputs = pars.top_k
device = pars.gpu
input_img = pars.input_img
checkpoint_path = pars.checkpoint
cat_names = pars.category_names
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

def loading_model_checkpoint ():
    checkpoint = torch.load (checkpoint_path)
    model =  getattr(models,checkpoint['arch'])(pretrained=True)

    if(checkpoint['arch']=='resnet50' ):
        model.fc=checkpoint ['classifier']
    elif(checkpoint['arch']=='vgg16' or checkpoint['arch']=='alexnet'):
        model.classifier=checkpoint ['classifier']
 
    
    model.load_state_dict (checkpoint ['model_state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 
        
    return model
model=loading_model_checkpoint()
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    img=Image.open(image_path)
    img.resize((256,256))
    pad = 0.5*(256-224)
    (left, upper, right, lower) = (pad, pad, 256-pad, 256-pad)
    img = img.crop( (left, upper, right, lower))
    np_img = np.array(img)/255
    np_img -= np.array([0.485, 0.456, 0.406])
    np_img /=np.array([0.229, 0.224, 0.225])

    return np_img.transpose((2,0,1))
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
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
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available()
    if cuda:
        # Move model parameters to the GPU
        model.cuda()
    else:
        model.cpu()

    #model.to(device)
    model.eval()
    im = process_image (image_path) 
    im = torch.from_numpy(np.array([im])).float()

    im = Variable(im)
    #im.to(device)
    if cuda:
        im = im.cuda()
    with torch.no_grad ():
        output = model.forward (im)
        
    output_prob = torch.exp (output) 
   
    
    probs, indeces = output_prob.topk (topk)
    prob = torch.topk(output_prob, topk)[0].tolist()[0]
    indeces = torch.topk(output_prob, topk)[1].tolist()[0]
    

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in indeces]

    classes = np.array (classes)
    
    
    return probs, classes

probs, classes = predict (path_image, model, number_of_outputs)

class_names = [cat_to_name [item] for item in classes]
for i in range(number_of_outputs):
    print(f"{class_names[i]} has probability of { probs[0][i]*100:.3f}%")