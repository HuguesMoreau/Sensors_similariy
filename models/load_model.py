"""
This file contains the necessary to reconstruct the intermediary featuress from
a save of the models an inputs

Author Hugues
"""

import torch
from pathlib import Path

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from param import data_path
file_location = Path(data_path) / Path('models')
from models.store_model_SHL import create_filename, Diagnostic_CNN
from models.store_model_CIFAR import Diagnostic_ResNet
# Diagnostic_ResNet and Diagnostic_CNN will be used for class loading

datasets = ["CIFAR_10", "SHL_2018"]
sensors = {"CIFAR_10":["CIFAR_10"],
           "SHL_2018":["Gyr_y", "Acc_norm", "Mag_norm"]}



n_trials = 3 *2


#%%
def load_data(file_location, dataset, sanity_check=False):
    """
    Loads the data and performs some verificaions on the ordering and performance

    Parameters
    ----------
    file_location (Path object or str): the absolute or reltive path to the
        .pickle objects
    dataset (str): either 'SHL_2018' or 'CIFAR_10'
    sanity_check (bool): if True, also loads the raw data an makes sure that we can
        recreate the predictions.
        Defaults to False

    Returns
    -------
    data: dict
        keys = sensor (ex "Acc_norm" or "CIFAR_10")
        values = dict
            keys = split ('train' or 'val')
            values = list of numpy arrays (n_samples, ...)
                one array per initialization (3*2 = 6 by default)
    models: dict
        keys = sensor (ex "Acc_norm" or "CIFAR_10")
        values = list of PyTorch nn.Module objects
    ground_truth: dict
        keys = split ('train' or 'val')
        values = np array of ints, containing the class between 0 and n-1
    """

    sensors_list = sensors[dataset]
    data =  {sensor:
                {split:
                    []
                for split in ["train", "val"]}
            for sensor in sensors_list}
    models = {sensor:
               []
            for sensor in sensors_list}
    ground_truth = {split:
            []
        for split in ["train", "val"]}
    if sanity_check: previous_GT = {"train":None, "val":None} # we will check that
        # the dataloader does not shuffle the position of the samples

    # basic sensors
    for sensor in sensors_list:
        if sanity_check:
            train_dataloader, val_dataloader = torch.load(Path(data_path) / Path("models") / Path("dataloaders-"+dataset+"-"+sensor+'.pt'))
            dataloaders = {'train':train_dataloader,
                           'val':  val_dataloader}

        for trial_index in range(n_trials):
            filename = create_filename(dataset, sensor, trial_index)
            features_filepath = Path(data_path) / Path("models") / Path('features-' + filename)
            model_filepath    = Path(data_path) / Path("models") / Path('model-' + filename)
            print(f"loading '{features_filepath}'...", end='')
            features_pred_GT_train, features_pred_GT_val = torch.load(features_filepath)
            model = torch.load(model_filepath)
            features_pred_GT = {"train":features_pred_GT_train,
                                "val"  :features_pred_GT_val
                                }
            print(' ... done')

            for i_split, split in enumerate(["train", "val"]):
                features, prediction, this_gt = features_pred_GT[split]
                ground_truth[split] = this_gt # the value is replaced every time, which is not
                     # a problem because all GT should be equal
                if sanity_check:
                    score_name, score_value = model.validate(dataloaders[split])
                    print(f"   {dataset:5s} {score_name} {100*score_value:.2f} %")
                    if previous_GT[split] is None:
                       previous_GT[split] = this_gt
                    else :
                        assert (previous_GT[split] == this_gt).all(), "the order of the samples changed between runs"
                data[sensor][split].append(features)

            model.cpu() # we dont need the model to be on GPU anymore
            models[sensor].append(model)

    return data, models, ground_truth



    #%%
if __name__ == "__main__":
    load_data(file_location, dataset="SHL_2018", sanity_check=True)







