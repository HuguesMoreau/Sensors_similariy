"""
Author Hugues

Evaluate the effectiveness of the features obtained by different feature
reduction algorithms.
This script is essentially a series of for loops, and it is long (a few hours)
"""
if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")
import torch
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from param import device
from models.store_model_SHL   import Diagnostic_CNN
from models.store_model_CIFAR import Diagnostic_ResNet # the classes are needed to load the pickle
from models.load_model import load_data, file_location
from models.load_model import sensors as sensors_per_dataset
from similarity.feature_reduction import FeatureReduction

methods_list = [ 'CCA_highest', 'CCA_random', 'CCA_lowest', 'PCA', 'max_activation', 'random_proj', 'random_keep']
random_methods_list = ['random_proj', 'random_keep', 'CCA_random']  # we will repreat these methods more often because of the randomness
n_repeat_random = 3 # repeat each random feature reduction 3 times per couple
    # of networks
n_repeat_FCLayer = 1 # we only repeat each training of the FC layer 1 time
    # because we already have 3 networks per sensor.
n_features_list = list(range(1,16)) + [16, 32, 64, 128]
n_trials = 3

# plot options
colors = {'random_proj':    [0.2,  0.23, 1.],
          'random_keep':    [0.2,  0.8,  1.],
          'max_activation': [0.8,  0.2,  1.],
          'PCA':            [0.05, 0.9,  0.05],
          'CCA_highest':    [1,  0.,   0.],
          'CCA_lowest':     [0.9,  0.9,  0.],
          'CCA_random':     [1.,   0.5,  0.]}
legend_types = [Line2D([0], [0], linestyle='-', color=colors[method]) for method in methods_list]
score_names = {"SHL_2018":"F1-score", "CIFAR_10":"accuracy"}


    #%%
if __name__ == "__main__":


    """  reminder:
    data: dict
        keys = sensor (ex "Acc_norm" or "CIFAR_10")
        values = dict
            keys = split ('train' or 'val')
            values = list of numpy arrays (n_samples, ...)
                one array per initialization (3*2 = 6 by default)
    """

    for dataset in ["CIFAR_10", "SHL_2018"]:
        sensors_list = sensors_per_dataset[dataset]
        results_retrained_dict = {sensor:
                                    {method:
                                        {i_trial:
                                            {n_features:
                                                []        # list of F1 scores
                                             for n_features in n_features_list}
                                         for i_trial in range(n_trials)}
                                     for method in methods_list}
                                 for sensor in sensors_list}
        results_old_dict = deepcopy(results_retrained_dict)
            # the architecture is the same, but not the content.

        for i_sensor, sensor in enumerate(sensors_list):
            data, models, GT =  load_data(file_location,  dataset)

            for method in methods_list:
                for i_trial in range(n_trials) :
                    X_train = data[sensor]['train'][i_trial]
                    X_val =   data[sensor]['val'][i_trial]
                    Y_train = GT['train'].reshape(-1)
                    Y_val =   GT[ 'val' ].reshape(-1)

                    n_train_samples = X_train.shape[0]
                    n_val_samples   =   X_val.shape[0]
                    old_shape = X_train.shape[1:] # get rid of the number of samples

                    X_train = X_train.reshape(n_train_samples, -1)
                    X_val =     X_val.reshape(  n_val_samples, -1)

                    # For the CCA: use an other instance of the same sensor
                    next_i_trial = i_trial + n_trials
                    X_train_next = data[sensor]['train'][next_i_trial]
                    X_train_next = X_train_next.reshape(X_train_next.shape[0], -1)

                    this_n_repeat = n_repeat_random if method in random_methods_list else n_repeat_FCLayer
                    # we will also repeat the classification using the old model N times,
                    # but it is not long compared to the rest

                    for i_repeat in range(this_n_repeat):
                        if method in random_methods_list or i_repeat == 0:
                            #compute the feature reduction the first time or recompute it if it is random
                            feature_reduction = FeatureReduction(method=method)
                            feature_reduction.fit(X_train, X_train_next)

                        for n_features_to_keep in n_features_list:
                            if n_features_to_keep <= X_train.shape[1]:
                                print(sensor, method, i_trial, i_repeat, n_features_to_keep)
                                X_train_projected = feature_reduction.transform(X_train, n_features_to_keep, same_dataset=True)
                                X_val_projected =   feature_reduction.transform(X_val,   n_features_to_keep, same_dataset=False)
                                X_train_projected   = X_train_projected.reshape(n_train_samples, *old_shape)
                                X_val_projected     =   X_val_projected.reshape(n_val_samples,   *old_shape)
                                old_model = models[sensor][i_trial]
                                model = deepcopy(old_model)
                                model.to(device)
                                model.eval()
                                model.forward = lambda X:model.forward_from_FC(X)

                                #using the old FC layer
                                val_dataset =   torch.utils.data.TensorDataset(torch.tensor(X_val_projected, dtype=torch.float32, device=device),
                                                                               torch.tensor(Y_val, dtype=torch.long, device=device))
                                val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, num_workers=0)

                                _, old_score = model.validate(val_dataloader)
                                results_old_dict[sensor][method][i_trial][n_features_to_keep].append(old_score)

                                # retrain the model
                                train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_projected, dtype=torch.float32, device=device),
                                                                                torch.tensor(Y_train, dtype=torch.long, device=device))
                                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True)


                                val_dataset =   torch.utils.data.TensorDataset(torch.tensor(X_val_projected, dtype=torch.float32, device=device),
                                                                                torch.tensor(Y_val, dtype=torch.long, device=device))
                                val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, num_workers=0)
                                model.to(device)
                                model.train_process(train_dataloader, val_dataloader)
                                _, retrained_score = model.validate(val_dataloader)
                                results_retrained_dict[sensor][method][i_trial][n_features_to_keep].append(retrained_score)

                            else :   # n_features_to_keep not in n_features_list:
                                if n_features_to_keep in results_retrained_dict[sensor][method][i_trial]:
                                    del results_retrained_dict[sensor][method][i_trial][n_features_to_keep]
                                    del results_old_dict[      sensor][method][i_trial][n_features_to_keep]


#%%
        # =============================================================================
        # Plot the results
        # =============================================================================

        # we are still in the first for loop (for dataset in ["CIFAR_10", "SHL_2018"])
        plt.figure(figsize=(len(sensors_list)*4, 8))
        matplotlib.rcParams['font.size'] = 10
        n_classes = np.max(GT["val"])+1

        for i_sensor, sensor in enumerate(sensors_list):
            for method in methods_list:
                mean_results_retrained = {} # results of the retrained FC layer
                mean_results_old_layer = {} # results of the stored FC layer
                std_results_retrained = {}
                std_results_old_layer = {}

                this_n_features_list = list(results_old_dict[sensor][method][0].keys())
                    #the list is assumed to be the same for all initializations of the net.
                for n_features in list(this_n_features_list):
                    results_old, results_retrained = [], []
                    for i_trial in range(n_trials):
                        results_old       +=       results_old_dict[sensor][method][i_trial][n_features]
                        results_retrained += results_retrained_dict[sensor][method][i_trial][n_features]
                    mean_results_retrained[n_features] = np.mean(results_retrained)
                    mean_results_old_layer[n_features] = np.mean(results_old)
                    std_results_retrained[n_features]  = np.std(results_retrained)
                    std_results_old_layer[n_features]  = np.std(results_old)

                mean_retrained = np.array( list(mean_results_retrained.values()))
                std_retrained =  np.array( list(std_results_retrained.values()))
                mean_old_model = np.array( list(mean_results_old_layer.values()))
                std_old_model =  np.array( list(std_results_old_layer.values()))


                plt.subplot(2, len(sensors_list), i_sensor+1)
                plt.title(sensor)
                plt.xscale('log')
                plt.grid('on')
                if i_sensor == 0: plt.ylabel("previous layer\n"+score_names[dataset])
                plt.plot(this_n_features_list, mean_old_model, color=colors[method])
                plt.fill_between(this_n_features_list, mean_old_model-std_old_model, mean_old_model+std_old_model, color=colors[method]+[0.1], linestyle='-')


                plt.subplot(2, len(sensors_list), i_sensor+1 +len(sensors_list))
                plt.xscale('log')
                plt.grid('on')
                if i_sensor == 0: plt.ylabel("retrained layer\n"+score_names[dataset])
                plt.plot(this_n_features_list, mean_retrained, color=colors[method])
                plt.fill_between(this_n_features_list, mean_retrained-std_retrained, mean_retrained+std_retrained, color=colors[method]+[0.1], linestyle='-')
                plt.xlabel("# components kept")

                if i_sensor == len(sensors_list)-1: plt.legend(legend_types, methods_list, fontsize=10, loc='lower right')


                if method == "CCA_highest":
                    plt.subplot(2, len(sensors_list), i_sensor+1)
                    plt.plot([n_classes, n_classes],
                             [0, mean_old_model.max()],
                             linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])
                    plt.plot([min(this_n_features_list), max(this_n_features_list)],
                             [mean_results_old_layer[n_classes], mean_results_old_layer[n_classes]],
                             linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])

                    plt.subplot(2, len(sensors_list), i_sensor+1 +len(sensors_list))
                    plt.plot([n_classes, n_classes],
                             [0, mean_retrained.max()],
                             linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])
                    plt.plot([min(this_n_features_list), max(this_n_features_list)],
                             [mean_results_retrained[n_classes], mean_results_retrained[n_classes]],
                             linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])


            plt.show()


            plt.savefig(f"{dataset}_fig.PNG")















