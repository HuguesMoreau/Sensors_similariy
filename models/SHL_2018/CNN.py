"""
Author Hugues

The CNN Defined here uses the architecture from

C. Ito, X. Cao, M. Shuzo, and E. Maeda, ‘Application of CNN for Human Activity
Recognition with FFT Spectrogram of Acceleration and Gyro Sensors’, in Proceedings
 of the 2018 ACM International Joint Conference and 2018 International Symposium
 on Pervasive and Ubiquitous Computing and Wearable Computers - UbiComp ’18, Singapore,
 Singapore, 2018, pp. 1503–1510. doi: 10.1145/3267305.3267517.

"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("../..")


import torch
import torch.nn as nn

from models.SHL_2018.base_network import Network





#%%
class CNN(Network):
    """
    A convolutional network that allows basic operations : early to late fusion,
    using depth-wise, time-wise, or channel-wise concatenation.
    Depending on the input_shape parameter, the convolutions are either 1D or 2D
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.mp = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()

        self.conv0 = nn.Conv2d( 1, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
        self.dropout0 = nn.Dropout(0.25)
        self.FC0 = nn.Linear(1600,128)
        self.dropout1 = nn.Dropout(0.5)
        self.FC1 = nn.Linear(128, 8)  # 8=nb of classes
        self.softmax = nn.Softmax(dim=1)



    def forward(self, X):
        """
        Input :
            X : an input tensor. Its shape should be equal to
                (B, input_shape[0] * n_branches, input_shape[1], input_shape[2])
        Output :
            scores : a (B, 8) tensor
        """
        X = self.conv0(X)
        X = self.mp(self.relu(X))
        X = self.conv1(X)
        X = self.mp(self.relu(X))
        X = self.conv2(X)
        X = self.mp(self.relu(X))
        X = X.view(X.shape[0],-1)
        X = self.dropout0(X)

        X = self.relu(self.FC0(X))

        X = self.dropout1(X)
        X = self.FC1(X)

        return X






#%%
if __name__ == "__main__":
    import torch.utils.data
    from models.SHL_2018 import Datasets
    from models.SHL_2018.transforms import SpectrogramTransform
    from models.SHL_2018.base_network import device

    transform = SpectrogramTransform("Acc_norm")
    train_dataset = Datasets.SignalsDataSet(mode='train', transform=transform)
    val_dataset =   Datasets.SignalsDataSet(mode='val',   transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, shuffle=False)

    model = CNN()
    model.to(device=device)
    model.train_process(train_dataloader, val_dataloader, maxepochs=50)
    model.plot_learning_curves()
    model.optim.zero_grad()
    model.plot_confusion_matrix(val_dataloader)


