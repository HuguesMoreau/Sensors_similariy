# -*- coding: utf-8 -*-
"""
Contains class that mimics the sklearn.decomposition.PCA class, with a fit()
and a transform() method. Here, this class proposes seven ways of reducing the
dimensionality
"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")
import numpy as np
from svcca import cca_core
from sklearn.decomposition import PCA
import random


class FeatureReduction():
    """ This class' method imitates roughly the sklearn.decomposition.PCA class.
    The major difference is that the number of components to keep is given at
    the very end, in the transform() function (sklearn.PCA asks the number of
    components at the creation of the object). The reason for this is that we
    will use the same reduction several times in a quick succession, and we want
    to avoid recomputing the CCA every time  """

    def __init__(self, method):
        """method is one of 'PCA', 'CCA_highest', 'CCA_lowest', 'CCA_random',
        'random_proj', 'random_keep', 'max_activation' """
        self.method=method


    def fit(self, X, Y=None):
        """
        This function does nothing when method = 'random_proj', 'random_keep',
        'max_activation'; it only computes the PCA and CCA when needed.

        In this case, it computes two kind of matrices:
        - base_change (np array with shape: (n_features, n_features))
        - inv_base_change (np array with shape: (n_features, n_features))
        These matrices are made in such a way that
            transform(X) = X @ base_change[:,:features] @ inv_base_change[:features,:]
        /!\ despite being called inv_base_change, the base_change is often not
        invertible, inv_base_change is only a pseudoinverse:
                   base_change   * inv_base_change !=   np.eye(n_features)
        but      inv_base_change *   base_change   == np.eye(n_features_to_keep)


       Parameters
        ----------
        - X (np array) input data. Shape: (n_samples, n_features)
        - Y (np array, optional). When using CCA, we need an other array to compute
            the new features. This array is discarded after CCA computation.
            is ignored when method is 'PCA' or 'random'

        Returns
        -------
        None

        """
        self.n_input_features = X.shape[1]
        self.mean = np.mean(X, axis=0, keepdims=True)

        if self.method == 'random_proj':
            # We generate a base change with rank equal to n_input_features, and
            # we will remove components to project on a subspace
            random_matrix = np.random.randn(self.n_input_features, self.n_input_features)
            complete_rotation, _ = np.linalg.qr(random_matrix)
            assert np.allclose(np.matmul(complete_rotation, complete_rotation.T), np.eye(self.n_input_features))

            self.base_change = complete_rotation
            self.inv_base_change = self.base_change.T

        elif self.method in ['random_keep', 'max_activation']:
            # for 'random_keep' and 'max_activation', the base change is only a
            # reordering of the identity matrix
            if self.method == 'random_keep' :
                index_features_sorted = list(range(self.n_input_features))
                random.shuffle(index_features_sorted)
            else :  #method == 'max_activation'
                max_activation_per_neuron = np.max(np.abs(X), axis=0) # shape: (n_features,)
                index_features_sorted = np.argsort(-max_activation_per_neuron)
                # argsort sorts from lowest to highest, we want the opposite
            self.base_change = np.eye(self.n_input_features)[:,index_features_sorted]
            self.inv_base_change = self.base_change.T


        elif self.method in ['PCA', 'CCA_highest', 'CCA_lowest', 'CCA_random']:
            n_components_pca = self.n_input_features if self.method == 'PCA' else (0.9999)
            # When performing a CCA, we anly want our matrix to have no zero singular values

            pca_X = PCA(svd_solver='full', n_components=n_components_pca)
            X_after_PCA = pca_X.fit_transform(X)
            self.pca_X  = pca_X
            self.X_after_PCA  = X_after_PCA  # keep the output value for debugging pruposes

            if self.method == 'PCA':
                self.base_change = pca_X.components_.T
                self.inv_base_change = pca_X.components_

            else : # CCA
                pca_Y = PCA(svd_solver='full', n_components=n_components_pca)
                Y_after_PCA = pca_Y.fit_transform(Y)
                self.pca_Y = pca_Y
                self.Y_after_PCA = Y_after_PCA  # keep the output value for debugging pruposes

                n_components_cca = min(pca_X.n_components_, pca_Y.n_components_)

                cca_results = cca_core.get_cca_similarity(X_after_PCA.T, Y_after_PCA.T, verbose=False,
                                          epsilon=1e-10, threshold=1-1e-6)


                self.X_after_CCA = X_after_PCA @ (cca_results["coef_x"] @ cca_results["full_invsqrt_xx"]).T  # shape: (n_samples, n_features)
                self.Y_after_CCA = Y_after_PCA @ (cca_results["coef_y"] @ cca_results["full_invsqrt_yy"]).T  # shape: (n_samples, n_features)

                self.cca = cca_results
                cca_base_changes = (cca_results["coef_x"] @ cca_results["full_invsqrt_xx"]).T
                inv_cca_base_changes = (np.linalg.inv(cca_results["full_invsqrt_xx"]) @ cca_results["full_coef_x"].T).T
                self.cca_base_changes = cca_base_changes

                self.base_change =     pca_X.components_.T  @ cca_base_changes
                self.inv_base_change = inv_cca_base_changes @ pca_X.components_

                if self.method == 'CCA_random':
                    index_features_sorted = list(range(min(self.n_input_features, n_components_cca)))
                    random.shuffle(index_features_sorted)
                    self.base_change =     self.base_change[:,index_features_sorted]
                    self.inv_base_change = self.inv_base_change[index_features_sorted,:]

                elif self.method == 'CCA_lowest':
                    self.base_change =     self.base_change[:,::-1]
                    self.inv_base_change = self.inv_base_change[::-1,:]

        else: # self.method does not fit in any of the other categories
            raise ValueError(f"Unknown method '{self.method}'")





    def transform(self, X, n_features_to_keep, same_dataset=False):
        """
        Parameters
        ----------
        - X (np array) input data. Shape: (n_samples, n_features)
        - n_features_to_keep (int)
        - same_dataset (bool, defaults to False):if True, tries to compare the
            result of pca.transform in self.fit with what we compute in this
            method, and raises an error if we cannot find the same results again.


        Returns
        -------
        - X_projected (np array with shape: (n_samples, n_features)): the array
            with reduced effective number of features, but projected back in the
            orginal coordinates (ie, rank(X_reduced) <= n_features_to_keep)
            Is homogeneous to the input matrix X
        """

        assert self.n_input_features == X.shape[1], f'the number of features changed (was {self.n_input_features} in self.fit, is {X.shape[1]})'

        if n_features_to_keep > self.n_input_features:
            raise ValueError(f'the number of features to keep ({n_features_to_keep}) must be inferior to the number of input features ({self.n_input_features})')
        assert n_features_to_keep > 0, f'We can only keep a strictly positive number of features (n_features_to_keep = {n_features_to_keep} was given)'

        error_message = f"X.shape={X.shape}, method='{self.method}', n_features_to_keep={n_features_to_keep}"
        # for the next assertions we will make here

        n_features_kept = self.base_change.shape[1]
        # when using CCA methods, the PCA sometimes leaves less components than asked (n_features_to_keep)

        assert np.allclose((self.inv_base_change @ self.base_change), np.eye(n_features_kept), atol=1e-4), error_message
                # Given the approximations of the computation, we need to have a high tolernce


        n_features_to_keep = min(n_features_kept, n_features_to_keep)

        X_reduced =     (X-self.mean) @ self.base_change[:,:n_features_to_keep]
        X_projected =   (X_reduced    @ self.inv_base_change[:n_features_to_keep,:]) + self.mean


        if same_dataset and self.method == 'PCA':
            assert np.allclose(self.X_after_PCA[:,:n_features_to_keep], X_reduced), error_message
            assert np.allclose(self.mean, self.pca_X.mean_), error_message

        return X_projected






if __name__ == "__main__":
    n_features = 120
    n_samples = 2000
    n_features_to_keep = 60
    methods_list = [ 'CCA_highest', 'CCA_random', 'CCA_lowest', 'PCA', 'max_activation', 'random_proj', 'random_keep']

    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # random_keep
    fr = FeatureReduction('random_keep')
    fr.fit(X)
    X_projected = fr.transform(X, n_features_to_keep, same_dataset=True)
    # make sure each column is either the mean or X
    X_equal_projected = np.isclose(X_projected, X)
    X_equal_mean =      np.isclose(X_projected,  np.mean(X, axis=0, keepdims=True))
    col_equal_projected = np.isclose(np.mean(X_equal_projected, axis=0), 1)
    col_equal_mean =      np.isclose(np.mean(X_equal_mean,      axis=0), 1)
    assert np.abs(np.mean(col_equal_mean+col_equal_projected, axis=0) -1) < 1e-6, 'columns must be equal to either the mean feature or X all the way'


    for method in methods_list:
        # if we keep all features, whatever the method, nothing should be lost (exce^t for CCA where we remove zero-variance axes)
        if method[:3] != 'CCA':
            fr = FeatureReduction(method)
            fr.fit(X, Y)
            X_projected = fr.transform(X, n_features_to_keep=n_features, same_dataset=True)
            error_message = f"method '{method}' should leave the data data intact when asked to keep all features"
            assert np.allclose(X, X_projected), error_message


        # some verifications are made inside the function, we want to run them
        # at least once with a non trivial amount of kept components
        fr = FeatureReduction(method)
        fr.fit(X, Y)
        X_projected = fr.transform(X, n_features_to_keep, same_dataset=True)
        result_rank = np.linalg.matrix_rank(X_projected -  np.mean(X, axis=0, keepdims=True))
        assert result_rank == n_features_to_keep, f"'{method}' returned a matrix wit rank {result_rank} (rank={n_features_to_keep} was expected)"
        # remark: in this case (random decorrelated gaussian noise), the PCA
        # will not remove any components, so the number of features kept is
        # exactly equal to the requested number

        # If we apply the reduction a second time, we sould not see any difference
        X_projected_twice = fr.transform(X_projected, n_features_to_keep, same_dataset=True)
        assert np.allclose(X_projected, X_projected_twice), f"All reductions should be idempotent. Reduction with '{method}' is not"

