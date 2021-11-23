"""
Author Hugues

Contains the utility toload the ordered data, split the train and validation
sets, and apply the reprocessing function

There are three ways to split the data:
    - 'shuffle': choose 13,000 samples at random, they will go in the training set,
        the other 3,310 go in the validation set. This is equivalent to doing
        like in the challenge, where most participant did not take the order
        into account. This also leads to overfitting, as fragments of a single
        trajectory can go in both sets, which means there is a possible
        contamination
    - 'unbalanced': the first 13,000 samples of the dataset (sorted chronologically)
        go in the training set, the other 3,310 go in the validation set.
        This is not ideal, for the sets now have different distributions.
        See visualizations/classes.py to understand the cause of the unbalance
    - 'balanced': the last 13,000 samples of the dataset (sorted chronologically)
        dataset go in the training set, the first 3,310 go in the validation set.
        This is the best way, for it produces sets with similar disributions

Note that this separation only applies to the training and validation sets,
the test set is kept the same as in th official challenge

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("../..")


import torch.utils.data
from timeit import default_timer as timer


import numpy as np

from pathlib import Path
from param import data_path, device
from models.SHL_2018.reorder import base_signals, segment_size

import pickle


len_train = 13000 # 13,000 elements in the train set





#%%

class SignalsDataSet(torch.utils.data.Dataset):
    def __init__(self, mode, transform=None):
        """
        Create a dataset for one mode (train, test, val)

        Parameters
        ----------
        - mode (string): either 'train', 'val', or 'test'
        - transform (function) : the function to apply to all samples (default: identity)
            It must take 2d arrays (shape: (_,6000)) as inputs, and return
            a numpy array as an output.
            Can be None (default)

        Returns
        -------
        a dataset object, containing a dict of numpy arrays
        """
        self.mode = mode
        self.transform = transform

        directory = "test" if mode == "test" else "train"
        file_path = Path(directory) / Path("ordered_data.pickle")
        self.path = file_path # useful for debugging

        complete_path = data_path / Path(file_path)
        print(f"\nDataset creation\n mode: {mode}")
        print(f" loading '{complete_path}'... ", end='')

        start = timer()
        with open(complete_path, "rb") as f:
            dict_data = pickle.load(f)
        end = timer()
        count = (end - start)
        print(f'load Data in {count:.3f} sec')

        len_file = dict_data['Label'].shape[0]
        len_val = len_file - len_train

        # ===================================================================
        #                         train/val split
        # =====================================================================


        if mode in ["train", "val"]:
            del dict_data['order']
            start = timer()
            # balanced splitting taking into account the chronological order
            # see https://github.com/HuguesMoreau/TMD_fusion_benchmark
            train_index = range(len_val, len_file)
            val_index   = range(0,       len_val)
            if mode == "train":
                chosen_index = train_index
            else: # mode == "val":
                chosen_index = val_index
            dict_data = {signal:dict_data[signal][chosen_index,:] for signal in dict_data.keys()}
            end = timer()
            count = (end - start)
            print('Split Train/Val in %.2f sec' % count)

        else: # test mode
            del dict_data['order']  # the order is not taken into account
                            # as we only use the test set for evaluation

        # =====================================================================
        #                    Apply preprocessing
        # =====================================================================
        start = timer()
        len_label_segment = dict_data["Label"].shape[1]  # take the label in the middle of the segment
        self.labels = torch.tensor(dict_data["Label"][:,len_label_segment//2], dtype=torch.long, device=device) # a Pytorch tensor on GPU
        del dict_data["Label"]
        print('before preprocessing:', end=' ')
        print({signal:dict_data[signal].shape for signal in dict_data})
        if self.transform != None:
            print("starting to apply preprocessing function:", self.transform)
            data = transform(dict_data)
            data = torch.tensor(data, dtype=torch.float32, device=device)
            print(f'after preprocessing: one tensor with shape {data.shape}')
            self.n_points = data.shape[1]
            self.len =      data.shape[0]     # number of samples

        else : # do not send the data to GPU, that is too much
            cpu = torch.device("cpu")
            dict_data = {signal_name:torch.tensor(dict_data[signal_name], dtype=torch.float32, device=cpu) for signal_name in dict_data.keys()}
            self.signals = list(dict_data.keys())
            matrix_example = dict_data[self.signals[0]] # to get the number of samples, we take the first matrix we have
            self.n_points = matrix_example.shape[1] # number of points per sample
            self.len = matrix_example.shape[0]     # number of samples
            print('after preprocessing: ', {signal:dict_data[signal].shape for signal in dict_data})
            assert (self.n_points == segment_size), "Inconsistency in the number of points per sample. Expected {}, got {}".format(self.n_points, segment_size)
             # reminder : segment_size comes from reorder.py

        end = timer()
        print(f'Preprocessing in {(end - start):.3f} sec')
        if self.transform == None:
            self.data = {signal:dict_data[signal] for signal in self.signals}
        else :
            self.data = data

    def __len__(self):
        return self.len

    def __getitem__(self,i):
        """
        Parameters
        ----------
        i is an integer between -self.len and self.len-1 (inclusive)


        Returns
        -------
        (selected_data, label)
        - selected_data is either a dictionnary with signals as keys, and
            torch.FloatTensor as values (shape: (1,6000) without preprocessing)
            or the result of the transform function given to the constructor
        - label is a torch.LongTensor with shape (1,)
        """
        if (i < -self.len or i >= self.len):
            raise IndexError("Incorrect index in '{}' dataset : {} (length of the dataset is {})".format(self.path, i, self.len))
        i = i%self.len # allows to use negative index
        label = self.labels[i] -1 #•self.labels are between 1 and 8, we want something between 0 and 1
        if self.transform != None:
            selected_data = self.data[i:i+1,...]
        else:
            selected_data = {signal:self.data[signal][i:i+1,:] for signal in self.signals}
        return (selected_data, label)






#%%
if __name__ == "__main__":
    """
    Print some samples from the original txt file, along with the same samples
    from two datasets (corresponding to different slits)
    """

    import time
    start_time = time.time()
    DS = SignalsDataSet(mode="train", transform=None)
    loading_time = time.time() - start_time
    print(f"data was successfully loaded in {loading_time:.3f} seconds")

    order_list = np.loadtxt(data_path / Path("train/train_order.txt")).astype(int)
    len_val = len(order_list) - len_train

    # Compare the first few elements of each file
    signal = "Gyr_z"
    filepath = data_path / Path("train/" + signal + ".txt")
    print("\n Comparisons betweeen files ({})".format(filepath))
    with open(filepath, "r") as f_signal:
        for i in range(10):
            order = order_list[i]
            data = next(f_signal)
            i_dataset = order -1 -len_val

            if order-1 > len_val:
                print("original txt file:         ", data[:30] + "..." + data[-30:-1])  # we do not print the last '\n'
                print("Dataset object    ", DS[i_dataset][0][signal])
                print('\n')

