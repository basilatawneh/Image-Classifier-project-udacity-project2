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


app = argparse.ArgumentParser(description='train.py')

app.add_argument('--data_dir',dest="data_dir" , action="store", default="./flowers/")
app.add_argument('--gpu', dest="gpu", action="store", default="cuda")
app.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
app.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
app.add_argument('--dropout', dest = "dropout", action = "store", default = 0.4)
app.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
app.add_argument('--arch', dest="arch", action="store", default="resnet50", type = str)
app.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
print(app.parse_args())
pars =  app.parse_args()
data_path = pars.data_dir
save_path = pars.save_dir
learning_rate = pars.learning_rate
model_structure = pars.arch
dropout_p = pars.dropout
hidden_layer1 = pars.hidden_units
device = pars.gpu
epochs = pars.epochs

models_allowed={
'resnet50':2048,
 'vgg16':25088,
    'alexnet':9216,
}
found = model_structure in models_allowed.keys()

if (model_structure in models_allowed.keys())==False:
    print('your model arch is invaled , choose from allowed models(resnet50, vgg16, alexnet)')
    exit()
train_dir = data_path + '/train'
valid_dir = data_path + '/valid'
test_dir = data_path + '/test'
data_transforms ={
    'train_transforms' :  transforms.Compose ([transforms.RandomRotation (30),
                                             transforms.RandomResizedCrop (224),
                                             transforms.RandomHorizontalFlip (),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ]),
   
    'valid_transforms' : transforms.Compose ([transforms.Resize (250),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) ])
    
} 

train_datasets = datasets.ImageFolder (train_dir, transform = data_transforms['train_transforms'])
valid_datasets = datasets.ImageFolder (valid_dir, transform = data_transforms['valid_transforms'])

train_images_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_images_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)

def model_genarator():
    model =  getattr(models,model_structure)(pretrained=True)
    in_features = models_allowed[model_structure]
    
    #Freeze feature parameters so as not to backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Build classifier for model
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_layer1)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout (p = dropout_p)),
                          ('fc2', nn.Linear(hidden_layer1, 512)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout (p = dropout_p)),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    criterion = nn.NLLLoss()
    if(model_structure=='resnet50' or model_structure == 'inception_v3'):
        model.fc=classifier
        optimizer = optim.Adam(model.fc.parameters(),lr=learning_rate)
    elif(model_structure=='vgg16' or model_structure=='alexnet'):
        model.classifier=classifier
        optimizer = optim.Adam(  model.classifier.parameters(),lr=learning_rate)

    return model, criterion, optimizer

model, criterion, optimizer = model_genarator()
print(model.classifier)
def train():
    model.to(device)
    step = 0
    print_every = 50

    for epoch in range(epochs):
        running_loss = 0
        for itr,(images,labels) in enumerate (train_images_loader):
            step = step+1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad ()
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % print_every == 0:
                test_loss = 0
                acc = 0
                model.eval()
                with torch.no_grad():
                    for Vimages, labels in valid_images_loader:
                        Vimages, labels = Vimages.to(device), labels.to(device)
                        log_ps = model.forward(Vimages)
                        batch_loss = criterion(log_ps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(valid_images_loader):.3f}.. "
                      f"Valid accuracy: {acc/len(valid_images_loader)*100:.3f}%")
                running_loss = 0
                model.train()
train()
checkpoint_data={
    'arch':model_structure,
    'hidden_units': hidden_layer1,
    'model_state_dict':model.state_dict(),
     'mapping':train_datasets.class_to_idx,
    'classifier': None
}
if(model_structure=='resnet50' or model_structure == 'inception_v3'):
        checkpoint_data['classifier']=model.fc
elif(model_structure=='vgg16' or model_structure=='alexnet'):
        checkpoint_data['classifier'] = model.classifier
torch.save (checkpoint_data,save_path)

print("The Model is trained")