import numpy as np

__all__ = ["est_t_matrix"]
def est_t_matrix(eta_corr, filter_outlier=False):

    # number of classes
    mum_classes = eta_corr.shape[1]
    T = np.empty((mum_classes, mum_classes))

    # find a 'perfect example' for each class
    for i in np.arange(mum_classes):

        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)

        for j in np.arange(mum_classes):
            T[i, j] = eta_corr[idx_best, j]

    return T