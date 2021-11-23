"""
Author Hugues

This script initializes neural networks, trains them, and records the features
in .pickle files
The naming syntax is defined in the 'create_filename' function

Many functions are only defined to provide a signature that is homogeneous to
other networks'
"""

import random
import os

import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

from models.SHL_2018.Datasets import SignalsDataSet
from models.SHL_2018.transforms import SpectrogramTransform
#from models.SHL_2018.fusion import ConcatCollate
#from architectures import basic_CNN, late_fusion
from models.SHL_2018.CNN import CNN
from param import device, classes_names, data_path

n_classes = len(classes_names)



class Diagnostic_CNN(CNN):
    """ Used to assign new methods to the networks without raising warnings
    when we reload a CNN classs that is different from what already exists """
    def __init__(self, *args, **kwargs):
        CNN.__init__(self, *args, **kwargs)
        self.classification_layer = self.FC1 # we need the classification layers
            #from all models to have the same name


    def train_process(self, train_dataloader, val_dataloader):
        """
        Override the base train_process method to remove the unwanted argument
        """
        return CNN.train_process(self, train_dataloader, val_dataloader, maxepochs=50)

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
        X = self.FC1(X)
        return X


    def record(self, dataloader):
        """
        Records all the hidden features and feature maps corresponding the the
        input samples in the dataloader, put them into a single dictionnary, and return it.

        Parameters
        ----------
        dataloader: a Pytorch Dataloader instance

        Returns
        -------
        a triple of :
            features:     afray of floats with shape (n_samples, n_features)
            predictions:  array of ints   with shape (n_samples,)
            ground_truth: array of ints   with shape (n_samples,)
        """
        features = np.zeros((len(dataloader.dataset), self.FC0.out_features))
        predictions = np.zeros((len(dataloader.dataset),))
        ground_truth = np.zeros((len(dataloader.dataset),))

        self.train(False)
        i_start = 0  # where to start inserting the data in the big arrays
                # i_end wil be updated every batch

        with torch.no_grad():
            for (X, Y) in dataloader:
                X_copy = X.clone()
                i_end = i_start + X.shape[0]   # = i_start + batch_size  (batch_size can change for the last batch)

                X = self.conv0(X)
                X = self.mp(self.relu(X))
                X = self.conv1(X)
                X = self.mp(self.relu(X))
                X = self.conv2(X)
                X = self.mp(self.relu(X))
                X = X.view(X.shape[0],-1)
                X = self.relu(self.FC0(X))
                features[i_start:i_end,:] = X.clone().detach().cpu().numpy()
                X  = self.FC1(X)

                predictions_this_batch = torch.argmax(X, dim=1)
                predictions[i_start:i_end] = predictions_this_batch.cpu().detach().numpy()
                ground_truth[i_start:i_end] = Y.cpu().detach().numpy()
                result_detailed = X
                assert(self(X_copy) == result_detailed).all()  # we check we did not forget any step

                self.optim.zero_grad()
                i_start = i_end

        return (features, predictions, ground_truth)



    def validate(self, dataloader):
        """
        Parameter: a Pytorch DataLoader
        Returns:
            score_name (string)
            score_value (float between 0 and 1)  """
        _, _, _, f1_score = CNN.train_process(self, [], dataloader, maxepochs=1)
        return ("f1_score", f1_score)



def create_filename(dataset, sensor_name, trial_index):
    return  f"{dataset}-{sensor_name}-{trial_index}.pt"



    #%%
if __name__ == "__main__":
    if 'models' not in os.listdir(data_path):
        os.mkdir(data_path/Path('models'))


    possible_sensors = ["Gyr_y", "Acc_norm", "Mag_norm"]
    n_repetitions = 3 *2 # 3 couples = 6 networks


    for sensor in possible_sensors:
        transform = SpectrogramTransform(sensor)
        train_dataset = SignalsDataSet(mode='train', transform=transform)
        val_dataset =   SignalsDataSet(mode='val',   transform=transform)

        train_dataloader_with_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        # it is usually better to train with shuffle (especially considering the data distribution),
        # but we do not want any shuffle when recording the data
        train_dataloader_no_shuffle   = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, shuffle=False)

        torch.save((train_dataloader_no_shuffle, val_dataloader),
                   Path(data_path) / Path("models") / Path(f'dataloaders-SHL_2018-{sensor}.pt'))

        for i_repetition in range(n_repetitions):
            model = Diagnostic_CNN()
            model.to(device)
            model.train_process(train_dataloader_with_shuffle, val_dataloader)
            filename = create_filename("SHL_2018", sensor, trial_index=i_repetition)
            torch.save(model, Path(data_path) / Path("models") / Path("model-" + filename))

            features_pred_GT_train = model.record(train_dataloader_no_shuffle)
            features_pred_GT_val   = model.record(val_dataloader)
            torch.save((features_pred_GT_train, features_pred_GT_val),
                       Path(data_path) / Path("models") /Path("features-" + filename))
                # note: the file type is named "feature" even though the file also
                # contains predictions and ground truth.

        del train_dataset, val_dataset, train_dataloader_no_shuffle, train_dataloader_with_shuffle, val_dataloader

