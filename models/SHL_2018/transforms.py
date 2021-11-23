#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains diverse preprocessing functions (mostly norms ans spectrograms),
and basic tests and visualizations.
If you are to work with any IPython console (ex: with Jupyter or spyder), is is advised
to launch a '%matplotlib qt' ,to get clean widow
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("../..")

import numpy as np
import torch
import scipy.signal, scipy.interpolate, scipy.ndimage


from param import classes_names, fs, duration_window, duration_overlap, spectro_batch_size
from models.SHL_2018 import Datasets

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_classes = len(classes_names)
    # We will need this for the tests
    DS = Datasets.SignalsDataSet(mode='train', transform=None)


#%% transform functions

"""In all following functions, the input parameter (data) is, by default,
 a dict of numpy arrays, containing signal names (eg. "Gyr_z") as keys, and 1-dimensional
 arrays as values

Most of this part contains basic visualizations to make sure the preprocessing is correct"""




class TemporalTransform():
    """  create the base transform to use to each element of the data

    Parameters
    ----------
    signal_name: a string (ex: 'Gyr_y', 'Ori_x')
        If the string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.

    Returns
    -------
    a function with input:  a dict of (_, 6000) arrays (key example: 'Gyr_y')
                and output: an array with the same shape.
    """
    def __init__(self, signal_name):
        super(TemporalTransform, self).__init__()
        self.signal_name = signal_name

    def __call__(self, data):
        """
        Parameters
        ----------
        data: a dict of (B, 6000) arrays (key example: 'Gyr_y')

        Returns
        -------
        an array with shape (B, 6000), where B depends on the input shape.
        """
        if self.signal_name[-2:] in ['_x', '_y', '_z', '_w'] or self.signal_name == "Pressure":
            processed_signal = data[self.signal_name]
        elif self.signal_name[-5:] == '_norm':
            suffix_location = self.signal_name.index("_") # 4 if signal_name == "LAcc", 3 otherwise
            sensor = self.signal_name[:suffix_location]    # ex: 'Acc', 'LAcc'
            if sensor == "Ori":
                # in that case, data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 should be 1.0
                processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 \
                                         + data[sensor+"_w"]**2)
            else :
                processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2)
        else :
            raise ValueError("unknown signal name: '{}'. Signal names should end with either '_x', '_y', '_z', '_w', or '_norm'".format(signal_name))
        return processed_signal



    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "Temporal_transform"
        str_to_return += f"\n\t Signal: {self.signal_name}"
        return str_to_return




if __name__ == "__main__":

    # plot one figure per sensor
    # on each figure, one subplot per class,
    # to find one instance per each class, we start looking at index = index0
    index0 = 0

    for tested_signal_name in ["Acc_norm", "Ori_norm", "Mag_norm", "LAcc_x"]:
        # plot 1 segment from each class.
        plt.figure()

        if tested_signal_name != 'Pressure':
            suffix_location = tested_signal_name.index("_")
            tested_sensor = tested_signal_name[:suffix_location]    # ex: 'Acc', 'LAcc'
        else:
            tested_sensor = 'Pressure'

        sensor_axis = [tested_sensor + axis for axis in ["_x", "_y", "_z"]] if tested_sensor != 'Pressure' else ['Pressure']
        if tested_sensor == "Ori" :  sensor_axis.append(tested_sensor+"_w")
        temporal_transform = TemporalTransform(tested_signal_name)
        remaining_classes = classes_names.copy()
        index = index0

        while len(remaining_classes)>0:
            data_tensor, class_tensor = DS[index] # data is a dict of 2D tensors (1,nb)
            data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}
            class_index = int(class_tensor)
            class_name = classes_names[class_index-1]

            if class_name in remaining_classes:

                remaining_classes.remove(class_name)
                plt.subplot(2, 4, n_classes - len(remaining_classes))
                for k,signal in enumerate(sensor_axis):

                    if k==0:  # compute the temporal axis once
                        nb = data_cpu[signal].shape[1]
                        x_t = np.linspace(0, nb/fs, nb)

                    plt.plot(x_t, data_cpu[signal][0,:])
                selected_signal = temporal_transform(data_cpu)
                error_message_dtype = "One of the signals does not have the correct type: {}, {} \n dtype should be float32, is actually {}".format(tested_signal_name, str(temporal_transform), selected_signal.dtype)
                assert (selected_signal.dtype == 'float32'), error_message_dtype

                plt.plot(x_t, selected_signal[0,:], '--')
                plt.xlabel("t (s)")
                legend = sensor_axis + [tested_signal_name+' (selected)']
                plt.legend(legend)
                plt.title("{} ({}, index={})".format(tested_sensor, classes_names[class_index-1], index))
            index +=1
        plt.show()




#%%

#  ----------------  Spectrogram transforms  ---------------------


# Interpolation functions
def interpol_log(f, t, spectrogram, out_size):
    """interpolates the spectrogram in input using a linear axis for the timestamps and a LOG axis for the frequencies

    Parameters
    ----------
    f : numpy array, shape: (F_in,), frequencies of the spectrogram
    t : numpy array, shape: (T_in,), timestamps of the spectrogram
    spectrogram : (B, F_in, T_in), B is batch size; 3D numpy array

    out_size : couple of ints (F_out, T_out)

    Returns
    -------
    f_interpolated : numpy array, shape: (F_out,), frequencies of the spectrogram AFTER interpolation
    t_interpolated : numpy array, shape: (T_out,), timestamps of the spectrogram AFTER interpolation
    a spectrogram, where the f axis (second dimension) has been re-interpolated
    using a log axis

    """
    B = spectrogram.shape[0]
    out_f, out_t = out_size

    log_f = np.log(f+f[1]) #  log between 0.2 Hz and 50.2 Hz

    log_f_normalized    = (log_f-log_f[0])/(log_f[-1]-log_f[0]) # between 0.0 and 1.0
    t_normalized        = (t-t[0])/(t[-1]-t[0])

    rescaled_f = out_f*log_f_normalized # 0 and 48
    # rescaled_f = (out_f-1)*log_f_normalized ??
    rescaled_t = out_t*t_normalized

    spectrogram_interpolated = np.zeros( (B, out_f, out_t), dtype='float32')
    index_f, index_t = np.arange(out_f), np.arange(out_t) # between 0 and 47

    for i in range(B):
        spectrogram_fn = scipy.interpolate.interp2d(rescaled_t, rescaled_f, spectrogram[i,:,:], copy=False)
        # interp2d returns a 2D function
        spectrogram_interpolated[i,:,:] = spectrogram_fn(index_t, index_f)  # care to the order

    f_fn = scipy.interpolate.interp1d(rescaled_f, f, copy=False)
    f_interpolated = f_fn(index_f)

    t_fn = scipy.interpolate.interp1d(rescaled_t, t, copy=False)
    t_interpolated = t_fn(index_t)


    return f_interpolated, t_interpolated, spectrogram_interpolated








#%%
#  ---------------- The spectrogram class --------------
class SpectrogramTransform():
    """ create the transform to work with spectrograms. This class behaves
    essentially the same as TempralTransform, except the created transform
    returns a dict of 3d array instead of 2d


    Parameters
    ----------
    signal_name: a string signal (ex: 'Gyr_y', 'Ori_x')
        If the string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.

    Returns
    -------
    a function with input: data : a dict of (_, 6000) arrays  (key example: 'Gyr_y')
                and output: a dictionnary of 2d arrays.

    """
    def __init__(self, signal_name):
        super(SpectrogramTransform, self).__init__()

        self.temporal_transform = TemporalTransform(signal_name)
        self.fs = fs
        self.duration_window = duration_window
        self.duration_overlap = duration_overlap
        self.spectro_batch_size = spectro_batch_size  # these values were loaded from the param file
        self.signal_name = signal_name
        self.out_size = (48, 48)

    def __call__(self, data):
        """
        Parameters
        ----------
        data : a dict of (B, 6000) arrays  (key example: 'Gyr_y')

        Returns
        -------
        An array with  shape (B, F, T), where B (dataset size) depends on the
            input shape, and F and T are equal to 48 here.
        """
        temporal_signal = self.temporal_transform(data)
        del data  # free some memory
        fs = self.fs
        nperseg     = int(self.duration_window * fs)
        noverlap    = int(self.duration_overlap * fs)

        spectro_batch_size = self.spectro_batch_size
        # turning 13,000 temporal signals into (550, 500) array
            # spectrograms at once is too much: a single (13000, 550, 500) array,
            # with simple precision requires 7.15 Go !
            # This is why we work with batches of 1000 instead. For each batch,
            # we compute the complete sectrogram (1000 x 550 x 500), then
            # interpolate it to smaller sizes, before working wit the following batch.

        current_spectro_batch_size = temporal_signal.shape[0]

        if current_spectro_batch_size < spectro_batch_size :
            f, t, spectrogram = scipy.signal.spectrogram(temporal_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
            f_interpolated, t_interpolated, interpolated_spectrogram = interpol_log(f, t, spectrogram, self.out_size)
                    # f, t, and possibly out_size will be ignored when the function does not need them
        else :
            n_batches = (current_spectro_batch_size-1)//spectro_batch_size +1
            nb_interp_f, nb_interp_t = self.out_size
            interpolated_spectrogram = np.zeros((current_spectro_batch_size, nb_interp_f, nb_interp_t), dtype='float32')
            for i in range(n_batches):
                i_min =   i   * spectro_batch_size
                i_max = (i+1) * spectro_batch_size  # does not matter if it goes beyond current_spectro_batch_size
                this_temporal_signal = temporal_signal[i_min:i_max,:]
                f, t, spectrogram = scipy.signal.spectrogram(this_temporal_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
                f_interpolated, t_interpolated, interpolated_spectrogram[i_min:i_max,:,:] = interpol_log(f, t, spectrogram, self.out_size)
        del temporal_signal
        np.log(interpolated_spectrogram + 1e-10, dtype='float32', out=interpolated_spectrogram) # in-place operation
        self.f_interpolated = f_interpolated
        self.t_interpolated = t_interpolated
        return interpolated_spectrogram



    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "Spectrogram transform"
        str_to_return += f"\n\t Signals: {self.signal_name}"
        str_to_return += f"\n\t Output size: {self.out_size}"
        str_to_return += f"\n\t Interpolation: log-interpolation"
        str_to_return +=  "\n\t Log-power"
        return str_to_return

# end of class SpectrogramTransform():



#%%
if __name__ == "__main__":
    fontdict = {'fontsize':10}
    n_ticks = 10

    # we plot the raw spectrogram and two interpolated spectrograms for the following classes
    selected_classes = ["Run", "Walk"]
    remaining_classes = selected_classes.copy()
    nsel = len(selected_classes)
    index = 3204  # where to tart the search
    plt.figure(figsize=(12,8))
    signal_name = "Acc_norm"
    temporal_transform    = TemporalTransform(signal_name)  # we will plot the result
    spectrogram_transform = SpectrogramTransform(signal_name)

    while len(remaining_classes)>0:
        data_tensor, class_tensor = DS[index]
        data_cpu = {signal:data_tensor[signal].cpu().detach().numpy() for signal in data_tensor.keys()}
        class_index = int(class_tensor)
        class_name = classes_names[class_index-1]

        if class_name in remaining_classes:
            remaining_classes.remove(class_name)
            i_class = nsel - len(remaining_classes)  # between 1 and n

            temporal_signal = temporal_transform(data_cpu)
            nb = temporal_signal.shape[1]
            x_t = np.linspace(0, nb/fs, nb)
            plt.subplot(2,nsel,i_class)
            plt.plot(x_t, temporal_signal[0,:])
            plt.title(f'{class_name} (index={index})', fontdict)
            plt.xlabel("t (sec)")
            plt.ylabel(signal_name)

            data_tensor, _ = DS[index]  # we need to recreate data because the variable is deleted
            data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}
            spectrogram_interpolated = spectrogram_transform(data_cpu)
            f_interpolated = spectrogram_transform.f_interpolated
            t_interpolated = spectrogram_transform.t_interpolated

            plt.subplot(2,nsel,i_class + nsel)
            t_interpolated = spectrogram_transform.t_interpolated
            f_interpolated = spectrogram_transform.f_interpolated
            matrix_shape = spectrogram_interpolated.shape
            time_list = [f'{t_interpolated[i]:.0f}' for i in np.round(np.linspace(0, matrix_shape[2]-1,n_ticks)).astype(int)]
            freq_list = [f'{f_interpolated[i]:.1f}' for i in np.round(np.linspace(0, matrix_shape[1]-1,n_ticks)).astype(int)]

            plt.xticks(np.linspace(0, matrix_shape[2]-1, n_ticks), time_list)
            plt.yticks(np.linspace(0, matrix_shape[1]-1, n_ticks), freq_list)
            plt.imshow(spectrogram_interpolated[0,:,:])

            plt.ylabel("f (Hz)")
            plt.xlabel("t (s)")
            plt.colorbar()

        index += 1

    plt.show()


#%%




