"""
Author Hugues

Evaluate the effectiveness of the features obtained by different feature
reduction algorithms. Here, we only consider the combination of two different
sensor data
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
from pathlib import Path
from param import data_path, device
from models.store_model_SHL   import Diagnostic_CNN
from models.store_model_CIFAR import Diagnostic_ResNet # the classes are needed to load the pickle
from models.load_model import load_data, file_location
from models.load_model import sensors as sensors_per_dataset
from similarity.feature_reduction import FeatureReduction
from similarity.projection_same_sensors import random_methods_list, n_repeat_random, n_repeat_FCLayer, \
            n_features_list, n_trials, colors, score_names
methods_list = ['CCA_highest', 'CCA_random', 'CCA_lowest'] # this time, we only consider the
    # projections that need an additional sensor. When the projection depends on only
    # one sensor, the results are the same as the ones in projection_same_sensors


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

    for dataset in ["SHL_2018"]:
        sensors_list = sensors_per_dataset[dataset]
        results_dict = {sensor_1:
                           {sensor_2:
                                {method:
                                    {i_trial:
                                        {n_features:
                                            []        # list of F1 scores
                                         for n_features in n_features_list}
                                     for i_trial in range(n_trials)}
                                 for method in methods_list}
                             for sensor_2 in sensors_list}
                         for sensor_1 in sensors_list}

        for i_sensor_1, sensor_1 in enumerate(sensors_list):
            for i_sensor_2, sensor_2 in enumerate(sensors_list):
                data, models, GT =  load_data(file_location,  dataset)

                for method in methods_list:
                    for i_trial in range(n_trials) :
                        X_train = data[sensor_1]['train'][i_trial]
                        X_val =   data[sensor_1]['val'][i_trial]
                        Y_train = GT['train'].reshape(-1)
                        Y_val =   GT[ 'val' ].reshape(-1)

                        n_train_samples = X_train.shape[0]
                        n_val_samples   =   X_val.shape[0]
                        old_shape = X_train.shape[1:] # get rid of the number of samples

                        X_train = X_train.reshape(n_train_samples, -1)
                        X_val =     X_val.reshape(  n_val_samples, -1)

                        # For the CCA: use an other instance of the same sensor
                        next_i_trial = i_trial + n_trials
                        X_train_next = data[sensor_2]['train'][next_i_trial]
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
                                    print(sensor_1, sensor_2, method, i_trial, i_repeat, n_features_to_keep)
                                    X_train_projected = feature_reduction.transform(X_train, n_features_to_keep, same_dataset=True)
                                    X_val_projected =   feature_reduction.transform(X_val,   n_features_to_keep, same_dataset=False)
                                    X_train_projected   = X_train_projected.reshape(n_train_samples, *old_shape)
                                    X_val_projected     =   X_val_projected.reshape(n_val_samples,   *old_shape)
                                    old_model = models[sensor_1][i_trial]
                                    model = deepcopy(old_model)
                                    model.to(device)
                                    model.eval()
                                    model.forward = lambda X:model.forward_from_FC(X)

                                    #using the old FC layer
                                    val_dataset =   torch.utils.data.TensorDataset(torch.tensor(X_val_projected, dtype=torch.float32, device=device),
                                                                                   torch.tensor(Y_val, dtype=torch.long, device=device))
                                    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, num_workers=0)

                                    _, old_score = model.validate(val_dataloader)
                                    results_dict[sensor_1][sensor_2][method][i_trial][n_features_to_keep].append(old_score)

        #%%
        # =============================================================================
        # Plot the results
        # =============================================================================

        # we are still in the first for loop (for dataset in ["SHL_2018"])
        n_sensors = len(sensors_list)
        plt.figure(figsize=(n_sensors*3.5, n_sensors*3))
        matplotlib.rcParams['font.size'] = 10
        n_classes = np.max(GT["val"])+1

        for i_sensor_1, sensor_1 in enumerate(sensors_list):
            for i_sensor_2, sensor_2 in enumerate(sensors_list):
                results_these_sensors = results_dict[sensor_1][sensor_2]

                plt.subplot(n_sensors, n_sensors, 1+i_sensor_1 + i_sensor_2*n_sensors)
                plt.grid('on')
                plt.xscale('log')
                if i_sensor_1 == 0: plt.ylabel(f"CCA with {sensor_2}\n {score_names[dataset]}")
                if i_sensor_2 == 0: plt.title(f"data from {sensor_1}")
                if i_sensor_2 == n_sensors-1: plt.xlabel("# components kept")
                if i_sensor_1 == n_sensors-1 and i_sensor_2 == n_sensors-1:
                    legend_types = [Line2D([0], [0], linestyle='-', color=colors[method]) for method in methods_list]
                    plt.legend(legend_types, methods_list, fontsize=10, loc='lower right')

                for method in methods_list:
                    results_per_n_features = {}

                    for i_trial in results_these_sensors[method].keys():
                        for n_features_to_keep in results_these_sensors[method][i_trial].keys():
                            if n_features_to_keep not in results_per_n_features:
                                results_per_n_features[n_features_to_keep] = []
                            results_per_n_features[n_features_to_keep] += results_these_sensors[method][i_trial][n_features_to_keep]

                    # reove empty elements
                    results_per_n_features = {k:v for (k,v) in results_per_n_features.items() if len(v) > 0}
                    n_features_list = list(results_per_n_features.keys())
                    mean = np.array([np.mean(results_per_n_features[n_f]) for n_f in results_per_n_features])
                    std  = np.array([ np.std(results_per_n_features[n_f]) for n_f in results_per_n_features])
                    plt.plot(n_features_list, mean, color=colors[method])
                    plt.fill_between(n_features_list, mean-std, mean+std, color=colors[method]+[0.1], linestyle='-')

                    if method == "CCA_highest":
                        plt.plot([n_classes, n_classes],
                                 [0, mean.max()],
                                 linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])
                        mean_n_classes = np.mean(results_per_n_features[n_classes])
                        plt.plot([min(n_features_list), max(n_features_list)],
                                 [mean_n_classes, mean_n_classes],
                                 linestyle='dotted', zorder=-10 , color=colors["CCA_highest"])


















