# adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm


DATA_DIR = "../tcga/dense256/"
VERBOSE = False
N_EPOCHS = 10

### SETUP CODE

def printConfusionMatrix(dc):
    print('__________________')
    print('|TP:' + str(dc['TP']) + " | FN: "+str(dc['FN']) + "|" )
    print('-----------------')
    print('|FP:' + str(dc['FP']) + " | TN: "+str(dc['TN']) + "|" )
    print('__________________')
    precision = dc['TP']  / float(dc['TP']+ dc['FP'])
    recall = dc['TP'] / float(dc['TP'] + dc['TN'])
    print('Precision: ' + str(precision) + " Recall: " + str(recall))
    print('__________________')

    
# trains the model, generates confusion matrices
def train_model(model, criterion, optimizer, scheduler, num_epochs=10, verbose=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss = []
    val_loss = []
    train_confusion = []
    val_confusion = []
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            train_loss = []
            val_loss = []
            running_loss = 0.0
            running_corrects = 0
            running_confusion = {"TP":0, "FP":0, "TN":0, "FN":0}
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_confusion['TP'] += torch.sum((labels.data == 1) * (preds == labels.data)).double()
                running_confusion['FP'] += torch.sum((labels.data == 1) * (preds != labels.data)).double()
                running_confusion['TN'] += torch.sum((labels.data == 0) * (preds == labels.data)).double()
                running_confusion['FN'] += torch.sum((labels.data == 0) * (preds != labels.data)).double()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #total_negs = dataset_sizes[phase]
            epoch_confusion = running_confusion
            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f} Caught: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, running_caught.double()))
            if phase == 'train':
                train_confusion.append(epoch_confusion)
                train_loss.append(running_loss)
            else:
                val_confusion.append(epoch_confusion)
                val_loss.append(running_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_confusion = epoch_confusion
                best_model_wts = copy.deepcopy(model.state_dict())
            
        if verbose:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    printConfusionMatrix(best_confusion)
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss, train_confusion, val_confusion, best_acc, best_confusion

def run_model(model, loss_fn, verbose, epochs=10):
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return train_model(model, loss_fn, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epochs, verbose=verbose)

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(num_images, num_images * 3))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, num_images //2, images_so_far)#num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
        
        
### RUN THE MODEL

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.1,0.1,0.1,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

squeeze1 = models.squeezenet1_1(pretrained=False, num_classes = 2)
ce_loss = nn.CrossEntropyLoss()

model, train_loss, val_loss, train_confusion, val_confusion, best_acc, best_confusion = run_model(squeeze1, ce_loss, verbose=False, epochs=N_EPOCHS)


