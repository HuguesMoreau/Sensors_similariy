"""
This file contains some functions to use better the ResNet, and to give them
methods that are similar to those of DiagnosticCNN
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import numpy as np
from pathlib import Path


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

from models.CIFAR_ResNet import ResNet, ResidualBlock, num_epochs, learning_rate
from param import device, data_path
from models.store_model_SHL import create_filename


class Diagnostic_ResNet(ResNet):
    def __init__(self, *args, **kwargs):
        ResNet.__init__(self, *args, **kwargs)
        self.classification_layer = self.fc  # we need the classification layers
            #from all models to have the same name

    def forward_from_FC(self, X):
        """
        This function is to replace the forward() method.
        Parameters
        ----------
        X: torch tensor wich size is (Batch, Features)

        Returns
        -------
        scores: torch tensor with shape (batch, n_classes) giving the scores for each class
        """
        X = self.fc(X)
        return X


    def train_process(self, train_dataloader, val_dataloader):
        """
        Train the network and returns the train and validation accuracies

        Parameters
        ----------
        train_dataloader (pytorch DataLoader object)
        val_dataloader (pytorch DataLoader object)

        Returns
        -------
        The scores at the last epoch: a couple of:
        train_acc (float, between 0. and 1.)
        val_acc (float, between 0. and 1.)
        """
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        total_step = len(train_dataloader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            correct, total = 0, 0
            self.train()
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (i+1) % 100 == 0:
                    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            train_accuracy = correct/total

            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr

            # Evaluation on the val dataloader
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_accuracy = correct/total
            print (f"Epoch [{epoch+1}/{num_epochs}], train acc:{100*train_accuracy:.3f}%,  val acc:{100*val_accuracy:.3f}% ")

        return train_accuracy, val_accuracy  # return the last values




    def record(self, dataloader):
        """
        Records all the hidden features and feature maps corresponding the the
        input samples in the dataloader, put them into a single dictionnary, and return it.

        Parameters
        ----------
        dataloader: a Pytorch Dataloader instance

        Returns
        -------
        dict_data: dictionnary with
            keys = layer name (str)
            values = np array
        """
        features = np.zeros((len(dataloader.dataset), self.classification_layer.in_features))

        conv, bn, relu, block0, block1, block2, AvgPool, fc = list(self.children())
        predictions = np.zeros((len(dataloader.dataset),))
        ground_truth = np.zeros((len(dataloader.dataset),))

        i_start = 0  # where to start inserting the data in the big arrays
                # i_end wil be updated every batch
        with torch.no_grad():
            for i, (X, Y) in enumerate(dataloader):
                X = X.cuda()
                X_copy = X.clone()

                i_end = i_start + X.shape[0]   # = i_start + batch_size  (batch_size can change for the last batch)
                X = conv(X)
                X = relu(bn(X))
                X = block0(X)
                X = block1(X)
                X = block2(X)
                X = F.avg_pool2d(X, X.shape[3])
                X = X.view(X.shape[0],-1)
                features[i_start:i_end,...] = X.clone().cpu().detach().numpy()

                X = fc(X)

                predictions_this_batch = torch.argmax(X, dim=1)
                predictions[i_start:i_end] = predictions_this_batch.cpu().detach().numpy()
                ground_truth[i_start:i_end] = Y.cpu().detach().numpy()

                result_detailed = X
                assert(self(X_copy) == result_detailed).all()  # we check we rdid not forget anything

                i_start = i_end

        return (features, predictions, ground_truth)




    def validate(self, dataloader):
        """
        Compute the accuracy on the input dataloader and return it. (the code actualy comes from the CIFAR-ResNet file)

        Parameters
        ----------
        dataloader : a Pytorch Dataloader object

        Returns
        -------
        score_name (string)   (we need it because we will use several models with a validate method)
        score_value (float between 0 and 1)
        """
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return "accuracy", correct/total



if __name__ == "__main__":
    if 'models' not in os.listdir(data_path):
        os.mkdir(data_path/Path('models'))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataloader_data_aug = torch.utils.data.DataLoader(  # with data augmentation
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        num_workers=0, pin_memory=True)
    transform = transforms.Compose([transforms.ToTensor(), normalize])  # get rid of data augmentation
    train_dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transform, download=True),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    val_dataloader  = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transform, download=True),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    torch.save((train_dataloader, val_dataloader), Path(data_path) / Path("models") / Path('dataloaders-CIFAR_10-CIFAR_10.pt'))

    n_repetitions = 3 *2
    for i_trial in range(n_repetitions):
        model = Diagnostic_ResNet(ResidualBlock, [6, 6, 6]).to(device)
        model.train_process(train_dataloader_data_aug, val_dataloader)
        filename = create_filename("CIFAR_10", "CIFAR_10", trial_index=i_trial)
        torch.save(model, Path(data_path) / Path("models") / Path("model-" + filename))

        features_pred_GT_train = model.record(train_dataloader)
        features_pred_GT_val   = model.record(val_dataloader)
        torch.save((features_pred_GT_train, features_pred_GT_val),
                   Path(data_path) / Path("models") /Path("features-" + filename))
            # note: the file type is named "feature" even though the file also
            # contains predictions and ground truth.


#
